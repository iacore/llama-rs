use std::{ffi::CString, ptr::null_mut};

use llama_sys::*;

fn main() {
    unsafe { unsafe_main() }
}

unsafe fn unsafe_main() {
    let mut lparams = llama_context_default_params();
    lparams.n_ctx = 512;
    lparams.n_parts = 2;
    lparams.seed = 1680773806;
    lparams.f16_kv = true;
    lparams.use_mlock = false;

    llama_init_from_file(
        CString::from_vec_unchecked(b"models/ggml-vicuna-13b-4bit.bin".to_vec()).as_ptr(),
        lparams,
    );
}
