#To fill out the gold standard hand-annotated datasert for BERT finetuning, I had GPT-2 generate more sentences for each category based on our examples
#and then I manually checked them in Excel. I adapted this code from code we used for Connected Politcs and debugged with Claude

#importing libraries
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import os
from datetime import datetime

#gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#prompting gpt2
def generate_statements(num_each=100, model_name="gpt2-large"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    #padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    #prompts per category
    traditional_antisemitism_prompt = """Generate examples of traditional antisemitism without mentioning Israel. These should contain classic antisemitic tropes or rhetoric.

Examples:
1. it was an act of piracy, you have shit for brains, you half-penised kike.
2. Trump Torches Globalist Cabal at Davos.
3. I think this is why the jewish freemasons are obsessed with death, cause the creator of this world is satan. The jews belive that lucifer IS light

Generate more statements containing traditional antisemitic rhetoric without mentioning Israel:
1. """

    israel_critique_prompt = """Generate legitimate criticisms of Israeli policies or actions without using antisemitic tropes or generalizing about Jewish people.

Examples:
1. President Isaac Herzog said that all citizens of Gaza are responsible. This is an ethnic retaliation being inflicted by Israel on a nation where half of the population are minors.
2. Israel won't let you hear the end of it as an excuse to why they're somehow justified in indiscriminately bombing residential apartment buildings...
3. We're at the point where the most extreme zionists just want the Palestinians to live in apartheid forever...

Generate more statements containing legitimate criticism of Israeli policies without antisemitism:
1. """

    antisemitic_israel_critique_prompt = """Generate examples of statements that criticize Israel while incorporating antisemitic tropes or rhetoric.

Examples:
1. I don't understand why Zionists & Israelis say "it's antisemitic to say jews are colonizers"...
2. Take for example the killing of 50 children in Waco Texas... Does it sound like Gaza?
3. Israeli Prime Minister Benjamin Netanyahu's father admitted that Sephardic Jews are the only ones with Israelite blood...

Generate more statements containing criticism of Israel with antisemitic tropes or rhetoric:
1. """
    
    prompts = {
        "traditional_antisemitism": traditional_antisemitism_prompt,
        "israel_critique": israel_critique_prompt,
        "antisemitic_israel_critique": antisemitic_israel_critique_prompt
    }
    
    results = {}
    
    for category, prompt in prompts.items():
        print(f"\nGenerating {num_each} statements for {category}...")
        statements = set()  #avoid duplicates
        
        #track iterations with maximum to avoid infinite loops
        iterations = 0
        max_iterations = num_each * 3 
        
        while len(statements) < num_each and iterations < max_iterations:
            iterations += 1
            
            #shuffling the parts of the prompt for hopefully better results
            current_prompt = prompt
            if iterations % 5 == 0:
                prompt_parts = prompt.split("Examples:\n")[0] + "Examples:\n"
                examples = prompt.split("Examples:\n")[1].split("Generate more")[0].strip().split("\n")
                random.shuffle(examples)
                example_text = "\n".join(examples)
                generation_part = "Generate more" + prompt.split("Generate more")[1]
                current_prompt = prompt_parts + example_text + "\n\n" + generation_part
            
            #generate output
            inputs = tokenizer(current_prompt, return_tensors="pt").to(device)
            
            output = model.generate(
                inputs.input_ids,
                max_length=len(inputs.input_ids[0]) + 300,
                temperature=0.8, #more diverse output
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            #decode
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            #extract generated part
            new_content = generated_text[len(current_prompt):]
            
            #process
            if "\n2. " in new_content:
                items = new_content.split("\n")
                for item in items:
                    if not item.strip() or not any(char.isalpha() for char in item):
                        continue
                    
                    #remove numbering
                    if ". " in item[:5]: 
                        clean_item = item.split(". ", 1)[1].strip()
                    else:
                        clean_item = item.strip()
                    statements.add(clean_item)
            
            #filter for length
            if len(statements) > num_each * 1.25:
                sorted_statements = sorted(statements, key=lambda s: abs(len(s.split()) - 15))
                statements = set(sorted_statements[:int(num_each * 1.1)])  #keep more than needed
        
        #final selection
        final_statements = list(statements)[:num_each]
        results[category] = final_statements
    
    #save to excel
    output_df = pd.DataFrame({
        "traditional_antisemitism": pd.Series(results.get("traditional_antisemitism", [])),
        "israel_critique": pd.Series(results.get("israel_critique", [])),
        "antisemitic_israel_critique": pd.Series(results.get("antisemitic_israel_critique", []))
    })
    
    #save to excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"generated_antisemitism_statements_{timestamp}.xlsx"
    output_df.to_excel(output_file, index=False)
    print(f"\nResults saved to '{output_file}'")
    
    return results

#use
if __name__ == "__main__":
    generate_statements(num_each=100, model_name="gpt2-large")