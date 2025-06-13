import json
import tiktoken

encoding = tiktoken.encoding_for_model("text-embedding-3-small")

input_file = "Jan25-Data.jsonl"
output_file = "Jan25-chunked.jsonl"
max_tokens = 300

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line_num, line in enumerate(f_in, 1):

        try:
            data = json.loads(line)
            url = data["url"]
            content = data["content"]
            
            tokens = encoding.encode(content)
            
            for i in range(0, len(tokens), max_tokens):
                chunk_tokens = tokens[i:i + max_tokens]
                chunk_text = encoding.decode(chunk_tokens)
                
                json.dump({"url": url, "content": chunk_text}, f_out, ensure_ascii=False)
                f_out.write("\n")

        except Exception as e:
            print(f"[Error on line {line_num}] {e}")
