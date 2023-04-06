pub use llama_sys as sys;

#[cfg(test)]
mod tests;

use std::{
    ffi::{c_float, c_int, CStr, CString},
    ptr::NonNull,
};

use derive_more::{Deref, DerefMut, From};

#[derive(Deref, DerefMut, From, Debug, Copy, Clone)]
pub struct ContextParams(pub sys::llama_context_params);

impl Default for ContextParams {
    fn default() -> Self {
        Self(unsafe { sys::llama_context_default_params() })
    }
}

#[derive(Deref, DerefMut, From, Debug)]
pub struct Context(pub NonNull<sys::llama_context>);

impl Context {
    pub fn init_from_file_cstr(path_model: &CStr, params: ContextParams) -> Option<Self> {
        let ptr = unsafe { sys::llama_init_from_file(path_model.as_ptr(), params.0) };
        if let Some(x) = NonNull::new(ptr) {
            Some(Self(x))
        } else {
            None
        }
    }
    pub fn init_from_file_str_unchecked(path_model: &str, params: ContextParams) -> Option<Self> {
        let from_vec_unchecked =
            unsafe { CString::from_vec_unchecked(path_model.as_bytes().to_vec()) };
        Self::init_from_file_cstr(&from_vec_unchecked, params)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            sys::llama_free(self.0.as_ptr());
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KvCache<'a> {
    pub data: &'a [u8],
    pub token_count: c_int,
}

impl Context {
    pub fn kv_cache<'a>(&'a mut self) -> KvCache<'a> {
        let ctx = self.0.as_ptr();
        KvCache {
            data: unsafe {
                std::slice::from_raw_parts(
                    sys::llama_get_kv_cache(ctx),
                    sys::llama_get_kv_cache_size(ctx),
                )
            },
            token_count: unsafe { sys::llama_get_kv_cache_token_count(ctx) },
        }
    }

    pub fn set_kv_cache(&mut self, x: KvCache) {
        let ctx = self.0.as_ptr();
        unsafe { sys::llama_set_kv_cache(ctx, x.data.as_ptr(), x.data.len(), x.token_count) }
    }
}

#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("opaque eval error (hidden from llama.cpp API)")]
pub struct EvalError;

impl Context {
    /// Run the llama inference to obtain the logits and probabilities for the next token.
    /// `tokens` is the provided batch of new tokens to process
    /// `n_past` is the number of tokens to use from previous eval calls
    pub fn eval(
        &mut self,
        tokens: &[Token],
        n_past: c_int,
        n_threads: c_int,
    ) -> Result<(), EvalError> {
        let ctx = self.0.as_ptr();
        unsafe {
            let tokens: &[sys::llama_token] = std::mem::transmute(tokens); // jank: not sure if this is valid usage
            if sys::llama_eval(ctx, tokens.as_ptr(), tokens.len() as i32, n_past, n_threads) == 0 {
                Ok(())
            } else {
                Err(EvalError)
            }
        }
    }
}

#[derive(Deref, DerefMut, From, Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Token(pub sys::llama_token);

impl Token {
    pub fn bos() -> Self {
        Self(unsafe { sys::llama_token_bos() })
    }
    pub fn eos() -> Self {
        Self(unsafe { sys::llama_token_eos() })
    }
}

/// Represents `tokenize` Error
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("buffer not big enough, need len>={0}")]

pub struct TooManyTokens(
    /// should have this many tokens
    pub usize,
);

impl Context {
    // Convert the provided text into tokens.
    // The tokens pointer must be large enough to hold the resulting tokens.
    pub fn tokenize<'a>(
        &self,
        text: &CStr,
        buffer: &'a mut [Token],
        add_bos: bool,
    ) -> Result<&'a mut [Token], TooManyTokens> {
        let ctx = self.0.as_ptr();
        let buffer: &mut [sys::llama_token] = unsafe { std::mem::transmute(buffer) }; // jank: not sure if this is valid usage
        let ret = unsafe {
            sys::llama_tokenize(
                ctx,
                text.as_ptr(),
                buffer.as_mut_ptr(),
                buffer.len() as c_int,
                add_bos,
            )
        };
        let buffer: &mut [Token] = unsafe { std::mem::transmute(buffer) }; // jank: not sure if this is valid usage
        if ret >= 0 {
            Ok(&mut buffer[0..(ret as usize)])
        } else {
            Err(TooManyTokens((-ret) as usize))
        }
    }
    pub fn n_vocab(&self) -> c_int {
        let ctx = self.0.as_ptr();
        unsafe { sys::llama_n_vocab(ctx) }
    }
    pub fn n_ctx(&self) -> c_int {
        let ctx = self.0.as_ptr();
        unsafe { sys::llama_n_ctx(ctx) }
    }
    pub fn n_embd(&self) -> c_int {
        let ctx = self.0.as_ptr();
        unsafe { sys::llama_n_embd(ctx) }
    }
    pub unsafe fn logits_mut_ptr(&mut self) -> *mut c_float {
        let ctx = self.0.as_ptr();
        sys::llama_get_logits(ctx)
    }
    pub unsafe fn embeddings_mut_ptr(&mut self) -> *mut c_float {
        let ctx = self.0.as_ptr();
        sys::llama_get_embeddings(ctx)
    }
    pub fn token_to_cstr(&self, token: Token) -> &CStr {
        let ctx = self.0.as_ptr();
        unsafe { CStr::from_ptr(sys::llama_token_to_str(ctx, token.0)) }
    }
    pub fn sample_top_p_top_k(
        &mut self,
        last_n_tokens: &[Token],
        top_k: c_int,
        top_p: f32,
        temp: f32,
        repeat_penalty: f32,
    ) -> Token {
        let ctx = self.0.as_ptr();
        unsafe {
            let last_n_tokens: &[sys::llama_token] = std::mem::transmute(last_n_tokens); // jank: not sure if this is valid usage
            Token(sys::llama_sample_top_p_top_k(
                ctx,
                last_n_tokens.as_ptr(),
                last_n_tokens.len() as c_int,
                top_k,
                top_p,
                temp,
                repeat_penalty,
            ))
        }
    }
    pub fn print_timings(&self) -> () {
        let ctx = self.0.as_ptr();
        unsafe { sys::llama_print_timings(ctx) }
    }
    pub fn reset_timings(&mut self) -> () {
        let ctx = self.0.as_ptr();
        unsafe { sys::llama_reset_timings(ctx) }
    }
}

pub unsafe fn system_info() -> &'static CStr {
    CStr::from_ptr(sys::llama_print_system_info())
}

// === API TODO ===

// author's words: not great API - very likely to change
// fn llama_model_quantize
