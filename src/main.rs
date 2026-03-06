// 注释说明，每一行的注释注释的下一行的代码！

//#![allow(unused)]
mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use std::time::Instant;

/// chats 对话
fn chats(
    max_len: usize,
    top_p: f32,
    top_k: u32,
    temperature: f32,
    penalty: f32,
){
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("half").join("chat");
    let llama = model::Llama::<half::bf16>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let mut cache = llama.new_cache();
    let mut user;
    let mut input: String;
    let mut output_ids;
    println!("chat start!\n-----------------------------------------------------------------------------");
    loop {
        user = String::new();
        println!("user:");
        std::io::stdin().read_line(&mut user).expect("you need to type some words into console!");
        if user.trim().eq("/exit") {
            println!("chat over!");
            break;
        }
        input = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user
        );
        let binding = tokenizer.encode(input, true).unwrap();
        let input_ids = binding.get_ids();
        (output_ids, cache) = llama.chat(input_ids, max_len, top_p, top_k, temperature,penalty, cache);
        println!("assistant:");
        println!("{}", tokenizer.decode(&output_ids, true).unwrap());
    }
}

fn main() {

} 

/// story 根据提示词来补全
fn story(
    max_len: usize,
    top_p: f32,
    top_k: u32,
    temperature: f32,
    penalty: f32,
){
    // 开始计时加载模型时间
    let start = Instant::now();
    let project_dir = env!("CARGO_MANIFEST_DIR");
    //let model_dir = PathBuf::from(project_dir).join("models").join("half").join("chat");
    //let model_dir = PathBuf::from(project_dir).join("models").join("story");
    //let model_dir = PathBuf::from(project_dir).join("models").join("half").join("story");
    let model_dir = PathBuf::from(project_dir).join("models").join("test");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    // 结束计时加载模型时间并输出
    let duration = start.elapsed();
    println!("加载模型花了{:?}时间",duration);
    // 开始计时获取输入字符串token_id 时间
    let start = Instant::now();
    // 提示词输入
    let input = "hello! my name is";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    // 结束计时获取输入字符串token_id 时间并输出
    let duration = start.elapsed();
    println!("获取输入字符串token_id花了{:?}时间",duration);
    // 开始计时计算所需时间
    let start = Instant::now();
    let output_ids = llama.generate(
        input_ids,
        max_len,
        top_p,
        top_k,
        temperature,
        penalty,
    );
    // 结束计时计算所需时间
    let duration = start.elapsed();
    println!("计算花了{:?}时间",duration);
    let output = tokenizer.decode(&output_ids,true).unwrap();
    println!("{}", output);
}

mod test{
    use std::time::Instant;

    use crate::{chats, story};

    #[test]
    fn test_chat(){
        chats(50, 0.9,50,0.7,1.15);
    }

    #[test]
    fn test_story(){
        let start = Instant::now();
        story(10,0.85,40,0.1,1.5);
        let duration = start.elapsed();
        println!("------------生成结束-----------------------------------------------------------------------------------------------");
        println!("⏱️  总耗时: {:?}", duration);
        println!("-------------------------------------------------------------------------------------------------------------------");
    }
}