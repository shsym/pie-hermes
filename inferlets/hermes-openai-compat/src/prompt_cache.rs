use crate::types::{
    PrefixEnsureRequest, PrefixEnsureResponse, PromptCacheCapabilitiesResponse,
    PromptCacheFeatures, PromptCacheStateInfo, PromptPrefixIdentity, PromptPrefixSlotSummary,
    PromptRenderContext,
};
use inferlet::forward::Forward;
use inferlet::{Context, Model};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct RegisteredPromptSlot {
    pub placement: String,
    pub text: String,
    pub content_hash: String,
}

#[derive(Debug, Clone)]
pub struct RegisteredPromptPrefix {
    pub handle: String,
    pub export_name: String,
    pub prefix_hash: String,
    pub render_context_hash: String,
    pub cache_epoch: String,
    pub prefix_text: String,
    pub prefix_tokens: Vec<u32>,
    pub kv_page_last_len: usize,
    pub slots: Vec<RegisteredPromptSlot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredPromptSlot {
    placement: String,
    text: String,
    content_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredPromptPrefix {
    handle: String,
    export_name: String,
    prefix_hash: String,
    render_context_hash: String,
    cache_epoch: String,
    prefix_text: String,
    prefix_tokens: Vec<u32>,
    kv_page_last_len: usize,
    slots: Vec<StoredPromptSlot>,
}

const STORE_PREFIX: &str = "openai_compat.prompt_cache";
const EPOCH_KEY: &str = "openai_compat.prompt_cache.epoch";

pub fn cache_epoch() -> String {
    if let Some(epoch) = inferlet::store_get(EPOCH_KEY) {
        if !epoch.is_empty() {
            return epoch;
        }
    }

    let epoch = format!("ep_{}", &sha256_hex(&inferlet::get_instance_id())[..16]);
    inferlet::store_set(EPOCH_KEY, &epoch);
    epoch
}

pub fn capabilities_response() -> PromptCacheCapabilitiesResponse {
    PromptCacheCapabilitiesResponse {
        object: "pie.prompt_cache.capabilities".to_string(),
        version: "1".to_string(),
        features: PromptCacheFeatures {
            registered_prefix_blocks: true,
            conditional_modules: false,
            discontiguous_positions: false,
            scaffolds: false,
        },
        cache: PromptCacheStateInfo {
            scope: "instance".to_string(),
            epoch: cache_epoch(),
        },
    }
}

pub async fn ensure_prefix(
    model: &Model,
    request: PrefixEnsureRequest,
) -> Result<PrefixEnsureResponse, String> {
    if request.prefix.slots.is_empty() {
        return Err("prefix.slots must not be empty".to_string());
    }

    let render_context_hash = hash_render_context(&request.render_context);

    let mut slots = Vec::with_capacity(request.prefix.slots.len());
    for slot in &request.prefix.slots {
        if slot.content.format != "text" {
            return Err(format!(
                "unsupported slot content format for {}: {}",
                slot.placement, slot.content.format
            ));
        }
        if slot.content.text.is_empty() {
            return Err(format!("slot {} has empty text", slot.placement));
        }
        slots.push(RegisteredPromptSlot {
            placement: slot.placement.clone(),
            text: slot.content.text.clone(),
            content_hash: format!("sha256:{}", sha256_hex(&slot.content.text)),
        });
    }

    let prefix_tokens = crate::prompt_render::format_registered_prefix_tokens(
        model,
        &request.prefix.slots,
        &request.render_context,
    )
    .await?;
    if prefix_tokens.is_empty() {
        return Err("prefix rendered to zero tokens".to_string());
    }

    let prefix_text = slots
        .iter()
        .map(|slot| slot.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let prefix_hash_input = serde_json::json!({
        "schema": request.schema,
        "prompt_mode": request.prefix.prompt_mode,
        "render_context_hash": render_context_hash,
        "slots": slots.iter().map(|slot| serde_json::json!({
            "placement": slot.placement,
            "content_hash": slot.content_hash,
        })).collect::<Vec<_>>(),
    });
    let prefix_hash = format!(
        "sha256:{}",
        sha256_hex(&serde_json::to_string(&prefix_hash_input).unwrap_or_default())
    );

    let handle = handle_for_prefix_hash(&prefix_hash);
    let cache_epoch = cache_epoch();
    let export_name = export_name_for(&cache_epoch, &handle);

    if let Some(existing) = load_prefix(&handle) {
        let queue = model.create_queue();
        if !queue.import_kv_pages(&existing.export_name).is_empty() {
            return Ok(build_ensure_response(&existing, "hit"));
        }
    }

    let mut ctx = model.create_context();
    ctx.fill_tokens(prefix_tokens.clone());
    ctx.flush().await;

    ctx.queue.export_kv_pages(&ctx.kv_pages, &export_name);

    let registered = RegisteredPromptPrefix {
        handle: handle.clone(),
        export_name,
        prefix_hash: prefix_hash.clone(),
        render_context_hash,
        cache_epoch,
        prefix_text,
        prefix_tokens,
        kv_page_last_len: ctx.get_kv_page_last_len(),
        slots,
    };

    save_prefix(&registered);

    // Also save as a PrefixCheckpoint so the session_kv path (Level 1.5)
    // can import this prefix for cross-session sharing. When a new session_id
    // arrives and Level 1 (session KV) misses, Level 1.5 imports the
    // checkpoint directly — token-level prefix match is the safety guard.
    crate::session_cache::save_prefix_checkpoint(&crate::session_cache::PrefixCheckpoint {
        export_name: registered.export_name.clone(),
        token_ids: registered.prefix_tokens.clone(),
        kv_page_last_len: registered.kv_page_last_len,
        content_hash: String::new(), // not used by Level 1.5
    });

    Ok(build_ensure_response(&registered, "created"))
}

pub fn get_prefix(handle: &str) -> Option<RegisteredPromptPrefix> {
    load_prefix(handle)
}

pub fn import_prefix(model: &Model, handle: &str) -> Option<Context> {
    let registered = get_prefix(handle)?;
    let queue = model.create_queue();
    let kv_pages = queue.import_kv_pages(&registered.export_name);
    if kv_pages.is_empty() {
        return None;
    }

    Some(Context::from_imported_state(
        model,
        kv_pages,
        registered.prefix_tokens.clone(),
        registered.kv_page_last_len,
    ))
}

fn build_ensure_response(prefix: &RegisteredPromptPrefix, status: &str) -> PrefixEnsureResponse {
    PrefixEnsureResponse {
        object: "pie.prompt_cache.prefix".to_string(),
        version: "1".to_string(),
        prefix_handle: prefix.handle.clone(),
        status: status.to_string(),
        cache: PromptCacheStateInfo {
            scope: "instance".to_string(),
            epoch: prefix.cache_epoch.clone(),
        },
        identity: PromptPrefixIdentity {
            prefix_hash: prefix.prefix_hash.clone(),
        },
        slots: prefix
            .slots
            .iter()
            .map(|slot| PromptPrefixSlotSummary {
                placement: slot.placement.clone(),
                content_hash: slot.content_hash.clone(),
            })
            .collect(),
    }
}

fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

fn handle_for_prefix_hash(prefix_hash: &str) -> String {
    let digest = prefix_hash.strip_prefix("sha256:").unwrap_or(prefix_hash);
    format!("pfx_{}", digest)
}

fn export_name_for(cache_epoch: &str, handle: &str) -> String {
    format!("prompt-cache:{}:{}", cache_epoch, handle)
}

fn store_key_for(handle: &str) -> String {
    format!("{}.prefix.{}", STORE_PREFIX, handle)
}

fn save_prefix(prefix: &RegisteredPromptPrefix) {
    let stored = StoredPromptPrefix {
        handle: prefix.handle.clone(),
        export_name: prefix.export_name.clone(),
        prefix_hash: prefix.prefix_hash.clone(),
        render_context_hash: prefix.render_context_hash.clone(),
        cache_epoch: prefix.cache_epoch.clone(),
        prefix_text: prefix.prefix_text.clone(),
        prefix_tokens: prefix.prefix_tokens.clone(),
        kv_page_last_len: prefix.kv_page_last_len,
        slots: prefix
            .slots
            .iter()
            .map(|slot| StoredPromptSlot {
                placement: slot.placement.clone(),
                text: slot.text.clone(),
                content_hash: slot.content_hash.clone(),
            })
            .collect(),
    };

    if let Ok(json) = serde_json::to_string(&stored) {
        inferlet::store_set(&store_key_for(&prefix.handle), &json);
    }
}

fn load_prefix(handle: &str) -> Option<RegisteredPromptPrefix> {
    let json = inferlet::store_get(&store_key_for(handle))?;
    let stored: StoredPromptPrefix = serde_json::from_str(&json).ok()?;
    Some(RegisteredPromptPrefix {
        handle: stored.handle,
        export_name: stored.export_name,
        prefix_hash: stored.prefix_hash,
        render_context_hash: stored.render_context_hash,
        cache_epoch: stored.cache_epoch,
        prefix_text: stored.prefix_text,
        prefix_tokens: stored.prefix_tokens,
        kv_page_last_len: stored.kv_page_last_len,
        slots: stored
            .slots
            .into_iter()
            .map(|slot| RegisteredPromptSlot {
                placement: slot.placement,
                text: slot.text,
                content_hash: slot.content_hash,
            })
            .collect(),
    })
}

fn hash_render_context(render_context: &Option<PromptRenderContext>) -> String {
    let normalized = serde_json::to_string(render_context).unwrap_or_default();
    format!("sha256:{}", sha256_hex(&normalized))
}
