

/// 构建llama3_prompt 方法
pub fn build_llama3_prompt(user_input: &str, is_first_turn: bool) -> String {
    let bos = if is_first_turn { "<|begin_of_text|>" } else { "" };
    format!(
        "{}<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        bos, user_input
    )
}

