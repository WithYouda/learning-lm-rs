mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;

use tch::kind;


#[allow(unused)]
fn story(
    max_len: usize,
    top_p: f32,
    top_k: u32,
    temperature: f32
){
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    print!("\n{}", input);
    let output_ids = llama.generate(
        input_ids,
        max_len,
        top_p,
        top_k,
        temperature,
    );
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}

fn chats(
    max_len: usize,
    top_p: f32,
    top_k: u32,
    temperature: f32
){
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
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
        (output_ids, cache) = llama.chat(input_ids, max_len, top_p, top_k, temperature, cache);
        println!("assistant:");
        println!("{}", tokenizer.decode(&output_ids, true).unwrap());
    }
}

fn main() {
    chats(
        100,
        5.,
        10,
        1.
    );
}
