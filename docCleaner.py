from langchain.text_splitter import MarkdownHeaderTextSplitter
import langchain_core.documents
import re

def extract_data_from_row(row:str) -> list[str]:
    row_data = row[1:-1].split('|')
    row_data = [data.strip() for data in row_data]
    return row_data

def generate_row_text(headers:list[str], values:list[str]) -> str:
    row_text = ""
    for header, value in zip(headers, values):
        row_text += f"{header}: {value},\n"
    row_text += '\n'
    return row_text

def word_len(chunk:str) -> int:
    chunk = chunk.replace(' ', '\n')
    words = chunk.split('\n')
    return len(words)

def split_string_by_tokens(content:str, token_limit:int) -> list[str]:
    # [content[0+i:token_limit+i] for i in range(0, len(content), token_limit)]
    split_content = []
    words = content.split(' ')
    if len(words) > token_limit:
        words_split = [words[0+i:token_limit+i] for i in range(0, len(words), token_limit)]
        for group in words_split:
            new_content = " ".join(group)
            split_content.append(new_content)
    else:
        split_content = [content]
    return split_content

def content_handler(content:str, token_limit:int) -> list[str]:
    final_page_content = []
    # Table Handler
    if content.startswith('|'):
        chunk = ""
        rows = content.split('\n')
        headers = extract_data_from_row(rows.pop(0))
        del rows[0]
        for row in rows:
            row_data = extract_data_from_row(row)
            row_text = generate_row_text(headers, row_data)
            if word_len(row_text) > token_limit:
                final_page_content.append(chunk)
                chunk = ""
                final_page_content.append(row_text[:token_limit])
            elif word_len(chunk) + word_len(row_text) > token_limit:
                final_page_content.append(chunk)
                chunk = ""
                chunk += row_text
            else:
                chunk += row_text
        final_page_content.append(chunk)
    # Image section handler
    elif content.startswith('!['):
        return []
    else:
        final_page_content = split_string_by_tokens(content, token_limit)
    
    return final_page_content

def header_cleaner(header:str) -> str:
    cleaned_header = re.sub(r'[0-9.]', '', header)
    cleaned_header = cleaned_header.strip()
    return cleaned_header

def chunkify_document(document:list[langchain_core.documents.base.Document], token_limit:int = 200) -> list[str]:
    final_chunkified_document = []

    for section in document:
        # major_heading = header_cleaner(section.metadata['major_heading']) if 'major_heading' in section.metadata else None
        # minor_heading = header_cleaner(section.metadata['minor_heading']) if 'minor_heading' in section.metadata else None
        # sub_heading = header_cleaner(section.metadata['sub_heading']) if 'sub_heading' in section.metadata else None
        major_heading = section.metadata.get('major_heading', '')
        minor_heading = section.metadata.get('minor_heading', '')
        sub_heading = section.metadata.get('sub_heading', '')
        page_content = section.page_content

        final_page_content = content_handler(page_content, token_limit)
        
        for content in final_page_content:
            if len(content) == 0:
                continue
            heading = ""
            if major_heading:
                heading += f"# {major_heading}"
            if minor_heading:
                heading += f" < ## {minor_heading}"
            if sub_heading:
                heading += f" < ### {sub_heading}"
            full_text = heading + '\n' + section.page_content.strip()

            # chunk = f"Major Heading: {major_heading},\nMinor Heading: {minor_heading},\nSub Heading: {sub_heading},\nContent: {content}\n"
            final_chunkified_document.append(full_text)
    
    return final_chunkified_document

def md_doc_reader(filepath:str, token_limit:int = 200) -> list[str]:
    markdown_text = open(filepath).read()

    # Define the Markdown splitter, along with headers for dividing the file
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[('#', 'major_heading'), ('##', 'minor_heading'), ('###', 'sub_heading')]
    )
    split_document = splitter.split_text(markdown_text) # Returns document split into sections
    chunkified_document = chunkify_document(document = split_document, token_limit = token_limit) # Returns list of chunks of document based on token limits

    return chunkified_document


def main():
    filepath = 'README.md'
    chunkified_document = md_doc_reader(filepath=filepath)
    for line in chunkified_document:
        print(line)

if __name__ == "__main__":
    main()