import re

# Path to your markdown file
markdown_file = "1._sidebar.md"

# Base URL for the site
base_url = "https://tds.s-anand.net/"

# Read markdown content from file
with open(markdown_file, "r", encoding="utf-8") as f:
    markdown_text = f.read()

# Find all .md links
md_links = re.findall(r'\((\.\./)?([a-zA-Z0-9\-]+\.md)\)', markdown_text)

# Create full URLs
full_urls = [f"{base_url}{filename}" for _, filename in md_links]

# Write output to a file
with open("urls.txt", "w", encoding="utf-8") as f:
    for url in full_urls:
        f.write(url + "\n")
