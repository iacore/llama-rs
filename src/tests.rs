use std::ffi::CString;

use llama_sys::*;

#[test]
fn load_model() {
    let mut lparams = crate::ContextParams::default();
    lparams.n_ctx = 512;
    lparams.n_parts = 2;
    lparams.seed = 1680773806;
    lparams.f16_kv = true;
    lparams.use_mlock = false;

    let model_path = unsafe { CString::from_vec_unchecked(b"models/ggml-vicuna-13b-4bit.bin".to_vec()) };

    _ = crate::Context::init_from_file_cstr(&model_path, lparams);
}
