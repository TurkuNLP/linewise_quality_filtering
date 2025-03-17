def line_quality_prompt(lines, vocab, language):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are an expert text classifier specializing in LLM training data. You will be given a passage from a document in {language} and your task is to classify each line of text based on its suitability for inclusion in a language model training dataset. High-quality content is clean, meaningful, well-structured, and useful for training language models. Low-quality content includes boilerplate elements (e.g., navigation menus, footers), non-linguistic content, formatting tags, placeholders like 'Lorem ipsum', and spammy, irrelevant, or toxic language.
    
**Instructions:**

1. **Line Identification and Separation**:
   - Each line starts with "Line X:" where X is the line number. Treat each "Line X:" as a single unit, regardless of length; do not split lines.
   - Lines are separated by newline characters (`\n`) and dashes (`------`). If there's no newline character, treat the entire text as a single line.

2. **Contextual Classification**:
   - Use the context of all lines when classifying each one, as they are sequential and from the same document.
   - For example, a line starting with a hyphen might be part of a list and should be classified as "Clean."

3. **Assigning Labels**:
   - Assign **exactly one label** to each line.
   - If the line is suitable for inclusion, label it **"Clean"**.
   - If not, assign a specific and descriptive label explaining why it's unsuitable.
   - **Prefer labels from the provided list**. Only create a new label (max three words) if absolutely necessary.
   - **Do not use vague labels** like "Low-Quality," "Bad," "Unsuitable," etc. Labels must be specific and descriptive.

4. **Focus on Linguistic Content**:
   - Retain valuable and diverse linguistic content suitable for language model pre-training, including natural language patterns, standard advertising copy, commercial language, and promotional content written in natural language.
   - Minor grammatical errors, typos, or small mistakes do not disqualify a line from being "Clean." Only exclude lines with pervasive errors that significantly hinder understanding.
   - Mild expletives and controversial opinions do not disqualify a line from being "Clean." Only exclude lines with blatantly hateful, harmful or toxic content.

5. **Output Format**:
   - Your output must have exactly the same number of lines as the input, matching each line number correctly.
   - Output only the line number followed by the label in JSON format.
   - Do not include any additional text or explanations.

**Guidelines for "Clean" Lines**:

Assign "Clean" to lines that:

- Represent natural language suitable for training language models.
- Include informal and non-standard language, grammatical errors, questions, partial sentences, and common online expressions.
- Contain standard advertising or commercial language in natural sentences.
- Have properly formatted titles, headings, and readable content, even with stylistic elements.
- Include minor in-text elements like email addresses, dates, or URLs within natural sentences.
- Are general promotional content written in natural language.

**Guidelines for Non-"Clean" Lines**:

Lines not classified as "Clean" need a specific and descriptive label. Examples include lines that:

- Contain blatantly hateful or harmful language. 
- Include disclaimers, copyright notices, or terms and conditions.
- Consist of menu items, login links, buttons, or navigation menus.
- Contain random characters, garbled text, or excessive symbols.
- Include programming code, HTML tags, or markup languages (when actual code or markup appears).
- Present keywords, tags, or similar data without sufficient context.
- Are spam-like content not suitable for training.
- Are **excessively** promotional without natural language structure (e.g., a list of product names and prices without sentences).

**Possible Labels for Non-"Clean" Lines**:

{vocab}

**Example Input:**

Line 1: Welcome to our website!
------
Line 2: Contact us at support@example.com.
------
Line 3: ***** $$$$$
------
Line 4: <div>Content</div>
------

**Example Output:**

{{
"Line 1": "Clean",
"Line 2": "Clean",
"Line 3": "Non-Linguistic Content",
"Line 4": "HTML Tags"
}}<|eot_id|><|start_header_id|>user<|end_header_id|>

**Now, classify the following lines:**

{lines}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def review_synonyms(group_name, synonyms):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You will be given a list of candidate words that might be synonyms. The words describe the contents of text documents.
Your task is to analyze the list and decide whether all words are sufficiently similar in meaning to form one coherent group and match the name of the group or if they should be divided into multiple groups. Please follow these instructions:

1. **Definition of Synonymy and Coherence:**
   - Words do not have to be perfectly synonymous, but they should share a significant overlap in meaning, usage, or connotation to be considered part of the same group.
   - A coherent group is one in which every word is contextually or semantically similar, even if subtle differences exist.

2. **Handling Ambiguity:**
   - Some words have multiple meanings. Use the most common sense or the sense that best fits with the other words in the list.
   - If a word has multiple plausible senses, choose the one that aligns with a potential group or consider it an outlier if no clear fit exists.

3. **Grouping Requirements:**
   - Group words such that each group is as large as possible without including words that do not fit together.
   - Avoid creating more groups than necessary. Only create a new group if one or more words clearly do not belong in an otherwise coherent group.
   - Outlier words should form their own individual groups if they do not match any of the other words.
   
4. **Naming the Groups:**
   - The name of the group should be one of the words in the group.
   - The name of the group should be central to the meaning of the group, i.e., all words in the group should be synonymous with the name. 

4. **Output Format:**
   - Present your results as a list of groups. Each group should be named.
   - Do not modify the words in any way. Each word must appear in exactly one group.
   - Your output should be in JSON format. Do not add any text or explanation outside the JSON object.
    - Example: {{"group_name": [<"word">,
                               <"word">,
                               <"word">,
                               ...],
                 "group_name": [<"word">,
                               <"word">,
                               <"word">,
                               ...]
                }}<|eot_id|><|start_header_id|>user<|end_header_id|>
                
Now, here are the possible synonyms:
Name: {group_name}
Words:
{synonyms}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def label_lines(lines):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are an expert text classifier specializing in LLM training data. Your task is to classify each line or passage of text based on its suitability for inclusion in a language model training dataset. High-quality content is clean, meaningful, well-structured, and useful for training language models. Low-quality content includes boilerplate elements (e.g., navigation menus, footers), non-linguistic symbols, formatting tags, placeholders like 'Lorem ipsum', and spammy, irrelevant, or toxic language.
    
**Instructions:**

1. **Line Identification and Separation**:
   - Each line starts with "Line X:" where X is the line number. Treat each "Line X:" as a single unit, regardless of length; do not split lines.
   - Lines are separated by newline characters (`\n`) and dashes (`------`). If there's no newline character, treat the entire text as a single line.

2. **Contextual Classification**:
   - Use the context of all lines when classifying each one, as they are sequential and from the same document.
   - For example, a line starting with a hyphen might be part of a list and should be classified as "Clean."

3. **Assigning Labels**:
   - Assign **exactly one label** to each line.
   - If the line is suitable for inclusion, label it **"Clean"**.
   - If not, assign a specific and descriptive label explaining why it's unsuitable.
   - **Prefer labels from the provided list**. Only create a new label (max three words) if absolutely necessary.
   - **Do not use vague labels** like "Low-Quality," "Bad," "Unsuitable," etc. Labels must be specific and descriptive.

4. **Focus on Linguistic Content**:
   - Retain valuable and diverse linguistic content suitable for language model pre-training, including natural language patterns, standard advertising copy, commercial language, and promotional content written in natural language.
   - Minor grammatical errors, typos, or small mistakes do not disqualify a line from being "Clean." Only exclude lines with pervasive errors that significantly hinder understanding.
   - Mild expletives and controversial opinions do not disqualify a line from being "Clean." Only exclude lines with blatantly hateful, harmful or toxic content.

5. **Output Format**:
   - Your output must have exactly the same number of lines as the input, matching each line number correctly.
   - Output only the line number followed by the label in JSON format.
   - Do not include any additional text or explanations.

**Guidelines for "Clean" Lines**:

Assign "Clean" to lines that:

- Represent natural language suitable for training language models.
- Include informal and non-standard language, grammatical errors, questions, partial sentences, and common online expressions.
- Contain standard advertising or commercial language in natural sentences.
- Have properly formatted titles, headings, and readable content, even with stylistic elements.
- Include minor in-text elements like email addresses, dates, or URLs within natural sentences.
- Are general promotional content written in natural language.

**Guidelines for Non-"Clean" Lines**:

Lines not classified as "Clean" need a specific and descriptive label. Examples include lines that:

- Contain blatantly hateful or harmful language. 
- Include disclaimers, copyright notices, or terms and conditions.
- Consist of menu items, login links, buttons, or navigation menus.
- Contain random characters, garbled text, or excessive symbols.
- Include programming code, HTML tags, or markup languages (when actual code or markup appears).
- Present keywords, tags, or similar data without sufficient context.
- Are spam-like content not suitable for training.
- Are **excessively** promotional without natural language structure (e.g., a list of product names and prices without sentences).

**Possible Labels for Non-"Clean" Lines**:

{vocab}

**Example Input:**

Line 1: Welcome to our website!
------
Line 2: Contact us at support@example.com.
------
Line 3: ***** $$$$$
------
Line 4: <div>Content</div>
------

**Example Output:**

{{
"Line 1": "Clean",
"Line 2": "Clean",
"Line 3": "Non-Linguistic Content",
"Line 4": "HTML Tags"
}}<|eot_id|><|start_header_id|>user<|end_header_id|>

**Now, classify the following lines:**

{lines}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def review_synonyms(group_name, synonyms):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You will be given a list of candidate words that might be synonyms. The words describe the contents of text documents.
Your task is to analyze the list and decide whether all words are sufficiently similar in meaning to form one coherent group and match the name of the group or if they should be divided into multiple groups. Please follow these instructions:

1. **Definition of Synonymy and Coherence:**
   - Words do not have to be perfectly synonymous, but they should share a significant overlap in meaning, usage, or connotation to be considered part of the same group.
   - A coherent group is one in which every word is contextually or semantically similar, even if subtle differences exist.

2. **Handling Ambiguity:**
   - Some words have multiple meanings. Use the most common sense or the sense that best fits with the other words in the list.
   - If a word has multiple plausible senses, choose the one that aligns with a potential group or consider it an outlier if no clear fit exists.

3. **Grouping Requirements:**
   - Group words such that each group is as large as possible without including words that do not fit together.
   - Avoid creating more groups than necessary. Only create a new group if one or more words clearly do not belong in an otherwise coherent group.
   - Outlier words should form their own individual groups if they do not match any of the other words.
   
4. **Naming the Groups:**
   - The name of the group should be one of the words in the group.
   - The name of the group should be central to the meaning of the group, i.e., all words in the group should be synonymous with the name. 

4. **Output Format:**
   - Present your results as a list of groups. Each group should be named.
   - Do not modify the words in any way. Each word must appear in exactly one group.
   - Your output should be in JSON format. Do not add any text or explanation outside the JSON object.
    - Example: {{"group_name": [<"word">,
                               <"word">,
                               <"word">,
                               ...],
                 "group_name": [<"word">,
                               <"word">,
                               <"word">,
                               ...]
                }}<|eot_id|><|start_header_id|>user<|end_header_id|>
                
Now, here are the possible synonyms:
Name: {group_name}
Words:
{synonyms}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
