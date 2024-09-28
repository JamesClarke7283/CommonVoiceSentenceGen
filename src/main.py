import os
import sys
import re
import csv
import math
import time
import asyncio
from tqdm import tqdm
from difflib import SequenceMatcher

from src.config import load_config, get_config_hash
from src.logger import logger
from src.models import init_db, ConfigModel, SentenceModel

from openai import OpenAI
import language_tool_python

from tortoise import Tortoise, run_async
from tortoise.exceptions import OperationalError

async def main():
    # Load configuration
    config = load_config()
    config_hash = get_config_hash()

    # Initialize the database
    await init_db()

    # Check for existing config hash in database
    existing_config = await ConfigModel.first()
    if existing_config:
        if existing_config.config_hash != config_hash:
            # Config has changed, reset database
            logger.info("Configuration has changed, resetting the database.")
            await Tortoise.close_connections()
            db_file_path = os.path.join(os.path.dirname(__file__), '..', 'gen-common-voice-data.db')
            if os.path.exists(db_file_path):
                os.remove(db_file_path)
            await init_db()
            # Save new config hash
            await ConfigModel.create(config_hash=config_hash)
    else:
        # No existing config, save current config hash
        await ConfigModel.create(config_hash=config_hash)

    # Get client settings
    gen_api_key = config['client_generation']['api_key']
    gen_base_url = config['client_generation']['base_url']
    gen_model_name = config['client_generation']['model_name']

    val_api_key = config['client_validation']['api_key']
    val_base_url = config['client_validation']['base_url']
    val_model_name = config['client_validation']['model_name']

    # Get domains list
    domains = config['domains']['list']

    # Get sampling settings
    total_sentences = config['sampling']['total_sentences']
    sample_percent = config['sampling']['sample_percent']

    # Calculate sentences to generate
    sentences_to_generate = int(total_sentences / sample_percent)

    # Initialize clients
    gen_client = OpenAI(
        api_key=gen_api_key,
        base_url=gen_base_url,
    )

    val_client = OpenAI(
        api_key=val_api_key,
        base_url=val_base_url,
    )

    # Initialize language tool
    tool = language_tool_python.LanguageTool('en-GB')

    # Read contribution guidelines
    with open(os.path.join(os.path.dirname(__file__), '..', 'contribution-guidelines.txt'), 'r', encoding='utf-8') as f:
        contribution_guidelines = f.read()

    # Define system prompt
    system_prompt = f"""
You are to generate English sentences that follow these guidelines:

{contribution_guidelines}

The generated sentences should be between 9 and 14 words long.

The generated sentences can optionally fall into the following categories/domains, or not fall into those categories (in that case, the domain is left blank):

{', '.join(domains)}
"""

    # Load sentences from database
    existing_sentences = await SentenceModel.all()
    unique_sentences = set(s.text for s in existing_sentences)

    # Initialize variables for progress tracking
    generated_sentences_count = len(unique_sentences)

    pbar_gen = tqdm(total=sentences_to_generate, initial=len(unique_sentences), desc='Generating valid sentences', unit='sentence')

    # Function to check if two sentences are similar
    def is_similar(s1, s2, threshold=0.7):
        return SequenceMatcher(None, s1, s2).ratio() > threshold

    # Function to check if the sentence meets all the criteria
    def is_valid_sentence(sentence):
        # Remove leading/trailing whitespace
        sentence = sentence.strip()

        # Length: The sentence should be between 9 and 14 words.
        word_count = len(sentence.split())
        if word_count < 9 or word_count > 14:
            return False, "Sentence does not meet the word count criteria (9-14 words)."

        # Spelling and Punctuation: The sentence must be spelled correctly.
        matches = tool.check(sentence)
        if matches:
            return False, "Sentence has spelling or grammar errors."

        # Numbers: There should ideally be no digits in the source text.
        if re.search(r'\d', sentence):
            return False, "Sentence contains digits."

        # Abbreviations and Acronyms: Avoid sentences with abbreviations or acronyms.
        if re.search(r'\b([A-Z]{2,})\b', sentence):
            return False, "Sentence contains abbreviations or acronyms."

        # Special Characters and Foreign Letters: Letters must be valid in English.
        if not all(ord(c) < 128 for c in sentence):
            return False, "Sentence contains special characters or foreign letters."

        return True, ""

    # Function to generate text using a given client
    def generate_text(client, prompt, system_prompt, temperature, max_tokens, model):
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        while True:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                )
                break
            except Exception as e:
                logger.error(f"Error during API call: {e}")
                logger.info("Waiting for 10 minutes before retrying...")
                time.sleep(600)

        generated_text = response.choices[0].message.content
        return generated_text.strip()

    # Function to perform final validation using the validation client
    def final_validate_sentence(sentence):
        validation_prompt = f"""
Check if the following sentence adheres to all the contribution guidelines below. Respond with 'Valid' if it adheres to all the guidelines, or 'Invalid' if it violates any of them.

Sentence: "{sentence}"

Guidelines:

{contribution_guidelines}

Is the sentence valid?
"""
        messages = [{'role': 'user', 'content': validation_prompt}]
        while True:
            try:
                response = val_client.chat.completions.create(
                    model=val_model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=15,
                    n=1,
                    stop=None,
                )
                break
            except Exception as e:
                logger.error(f"Error during final validation: {e}")
                logger.info("Waiting for 10 minutes before retrying final validation due to error.")
                time.sleep(600)
        generated_text = response.choices[0].message.content.strip()
        return 'Valid' in generated_text

    print("Starting sentence generation...")

    # Generate sentences until we have enough
    while len(unique_sentences) < sentences_to_generate:
        generated_text = generate_text(
            client=gen_client,
            prompt='Generate an English sentence between 9 and 14 words long.',
            system_prompt=system_prompt,
            temperature=0.8,
            max_tokens=70,
            model=gen_model_name
        )

        # Split into sentences if multiple sentences are returned
        sentences = re.split(r'(?<=[.!?])\s+', generated_text)

        for sentence in sentences:
            sentence = sentence.strip()
            is_valid, reason = is_valid_sentence(sentence)
            if is_valid:
                # Check for duplicates and similarities
                if any(is_similar(sentence, s) for s in unique_sentences):
                    continue  # Skip if similar to an existing sentence
                unique_sentences.add(sentence)

                # Assign domains
                domains_list_str = '\n'.join(f"- {domain}" for domain in domains)
                domain_prompt = f"""Given the following sentence:

"{sentence}"

Identify up to 3 domains from the following list that the sentence falls into. Only include a domain if you are sure it applies. Be conservative.

Domains:
{domains_list_str}

Provide the domains as a comma-separated list. If none apply, respond with "None":
"""

                domain_text = generate_text(
                    client=gen_client,
                    prompt=domain_prompt,
                    system_prompt='',
                    temperature=0.3,
                    max_tokens=50,
                    model=gen_model_name
                )

                # If the response is "None", assign an empty list
                if domain_text.strip().lower() == 'none':
                    assigned_domains = []
                else:
                    # Extract domains from the response
                    assigned_domains = [domain.strip() for domain in re.split(r',\s*', domain_text) if domain.strip() in domains]
                    # Limit to up to 3 domains
                    assigned_domains = assigned_domains[:3]

                # Score the sentence
                score_prompt = f"""Rate the quality of the following sentence on a scale from 1 to 10, where 10 is the best. Only provide the number as the answer.

"{sentence}"

Quality score:"""

                score_text = generate_text(
                    client=gen_client,
                    prompt=score_prompt,
                    system_prompt='',
                    temperature=0.5,
                    max_tokens=5,
                    model=gen_model_name
                )

                # Extract the score
                match = re.search(r'\b([1-9]|10)\b', score_text)
                if match:
                    score = int(match.group(1))
                else:
                    score = 0  # Default score if unable to parse

                # Save the sentence to database
                await SentenceModel.create(
                    text=sentence,
                    is_validated=False,
                    domains=assigned_domains,
                    score=score,
                    is_final_validated=False
                )

                pbar_gen.update(1)
                logger.info(f"Generated {len(unique_sentences)}/{sentences_to_generate} valid sentences.")

                if len(unique_sentences) >= sentences_to_generate:
                    break
            else:
                continue

    pbar_gen.close()
    print("Sentence generation completed.")

    # Retrieve sentences from database for further processing
    all_sentences = await SentenceModel.all()

    # Sort sentences by score in descending order
    sorted_sentences = sorted(all_sentences, key=lambda x: x.score if x.score is not None else 0, reverse=True)

    # Select top sample_percent of sentences based on score
    top_count = max(1, int(len(sorted_sentences) * sample_percent))
    top_sentences = sorted_sentences[:top_count]

    # Ensure diversity across domains when selecting final sentences
    print("Selecting diverse sentences...")
    final_sentences = []
    domains_covered = set()
    used_sentences = set()

    for sentence_obj in top_sentences:
        if len(final_sentences) >= total_sentences:
            break

        sentence_domains = sentence_obj.domains
        # Skip sentences without domains
        if not sentence_domains:
            continue

        # If the sentence covers new domains, add it
        if any(domain not in domains_covered for domain in sentence_domains):
            final_sentences.append(sentence_obj)
            domains_covered.update(sentence_domains)
            used_sentences.add(sentence_obj.text)
        else:
            # If sentence is not too similar to already selected sentences
            if not any(is_similar(sentence_obj.text, s.text) for s in final_sentences):
                final_sentences.append(sentence_obj)
                used_sentences.add(sentence_obj.text)

    # If we still don't have enough sentences, fill up with sentences without domains or that are less similar
    if len(final_sentences) < total_sentences:
        for sentence_obj in top_sentences:
            if len(final_sentences) >= total_sentences:
                break
            if sentence_obj.text not in used_sentences:
                if not any(is_similar(sentence_obj.text, s.text) for s in final_sentences):
                    final_sentences.append(sentence_obj)
                    used_sentences.add(sentence_obj.text)

    print("Starting final validation with the superior model...")
    pbar_final_val = tqdm(total=len(final_sentences), desc='Final validation', unit='sentence')

    validated_sentences = []
    for idx, sentence_obj in enumerate(final_sentences):
        # Check if already validated
        if sentence_obj.is_final_validated:
            validated_sentences.append(sentence_obj)
            pbar_final_val.update(1)
            continue

        # Perform final validation
        try:
            if final_validate_sentence(sentence_obj.text):
                sentence_obj.is_final_validated = True
                await sentence_obj.save()
                validated_sentences.append(sentence_obj)
                # Do not print or log validation here to avoid logs on console
            else:
                logger.info(f"Sentence '{sentence_obj.text}' failed final validation.")
        except Exception as e:
            logger.error(f"Error during final validation: {e}")
            logger.info("Waiting for 10 minutes before retrying final validation due to error.")
            time.sleep(600)
            continue
        pbar_final_val.set_postfix({'Validated': f"{len(validated_sentences)}/{total_sentences}"})
        pbar_final_val.update(1)

        if len(validated_sentences) >= total_sentences:
            break

    pbar_final_val.close()
    print("Final validation completed.")

    if len(validated_sentences) < total_sentences:
        print("Not enough sentences passed final validation. Continuing generation...")
        # You can implement additional logic to generate more sentences here
    else:
        print("Required number of sentences validated.")

    # Write the validated sentences to a TSV file
    output_file = os.path.join(os.path.dirname(__file__), '..', 'sentences.tsv')
    with open(output_file, 'w', newline='', encoding='utf-8') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')

        # Write the header
        writer.writerow([
            'Sentence (mandatory)',
            'Source (mandatory)',
            'Additional rationale for open license (mandatory)',
            'Sentence Quality Assurance Feedback: leave blank, for internal use',
            'Domain (optional)',
            'Notes'  # Add a notes column
        ])

        # Process notes for each sentence
        process_notes = f"""
- {gen_model_name} is used to generate sentences and tag data.
- We generate {sentences_to_generate} sentences and pick the top {int(sample_percent * 100)}% ({total_sentences} sentences) based on quality score.
- We ensure sentences are between 9 and 14 words and check for similarity to promote diversity.
- We tag each sentence into a category/domain only if it fits into it.
- At the very end, we use a superior model ({val_model_name}) to ensure every sentence meets the guidelines before publication.
- Source Code: https://github.com/JamesClarke7283/CommonVoiceSentenceGen
""".strip()

        # Write each sentence with the required metadata
        for sentence_obj in validated_sentences[:total_sentences]:
            domain_field = ', '.join(sentence_obj.domains) if sentence_obj.domains else ''
            writer.writerow([
                sentence_obj.text,
                f'{gen_model_name}; data generated by it is used under appropriate licensing.',
                'Data is licensed as CC0.',
                '',  # Sentence Quality Assurance Feedback left blank
                domain_field,  # 'Domain' field
                process_notes
            ])

    print(f"{len(validated_sentences)} sentences have been generated and saved to 'sentences.tsv'.")

    logger.info(f"{len(validated_sentences)} sentences have been generated and saved to 'sentences.tsv'.")

    # Close the Tortoise ORM connections
    await Tortoise.close_connections()

# Entry point for script
if __name__ == '__main__':
    run_async(main())
