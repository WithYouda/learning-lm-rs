

/// 构建llama3_prompt 方法
pub fn build_llama3_prompt(user_input: &str, is_first_turn: bool) -> String {
    let bos = if is_first_turn { "<|begin_of_text|>" } else { "" };
    format!(
        "{}<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        bos, user_input
    )
}

/// 构建带有系统提示词的llama3_prompt
/// 仅在第一轮对话且system_prompt非空时插入系统提示词
pub fn build_llama3_prompt_with_system(user_input: &str, system_prompt: &str, is_first_turn: bool) -> String {
    let bos = if is_first_turn { "<|begin_of_text|>" } else { "" };
    if is_first_turn && !system_prompt.is_empty() {
        format!(
            "{}<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            bos, system_prompt, user_input
        )
    } else {
        format!(
            "{}<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            bos, user_input
        )
    }
}

