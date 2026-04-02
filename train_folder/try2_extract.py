import pandas as pd
import re
import os
from tqdm import tqdm
from langdetect import detect
from deep_translator import GoogleTranslator

cwd = "/Users/abhayaggarwal/Downloads/deep-past-initiative-machine-translation"
train_folder = os.path.join(cwd, "train_folder")

print("STEP 1: Starting Reference Build...")
published = pd.read_csv(os.path.join(cwd, "published_texts.csv"))
alias_map = {}
akkadian_reference_list = []

for _, row in published.iterrows():
    translit = str(row['transliteration']).lower() if pd.notna(row['transliteration']) else ""
    alias_str = str(row['aliases']) if pd.notna(row['aliases']) else ""
    if not translit.strip(): continue
    
    aliases = [x.strip() for x in alias_str.split('|') if x.strip()]
    for al in aliases:
        if len(al) > 2: alias_map[al] = translit
    akkadian_reference_list.append(translit)

reference_lines = []
for ref in akkadian_reference_list:
    reference_lines.extend([x.strip() for x in re.split(r'[\n\.]', ref) if len(x.strip()) > 5])
reference_lines = list(set(reference_lines))
print(f"Loaded {len(reference_lines)} reference lines.")

print("STEP 2: Filter Relevant Pages")
publications = pd.read_csv(os.path.join(cwd, "publications.csv"))
if 'has_akkadian' in publications.columns:
    publications = publications[publications['has_akkadian'] == True]

def clean_text(t):
    return re.sub(r'\s+', ' ', str(t).lower()).strip()

extracted_pairs = []
translator = GoogleTranslator(source='auto', target='en')

def translate_to_english(text):
    try:
        lang = detect(text)
        if lang == 'en': return text, lang
        return translator.translate(text), lang
    except:
        return text, 'unknown'

def is_akkadian_like(line):
    if re.search(r'[šṣṭḫāīū]', line): return True
    if re.search(r'\b[a-z]+-[a-z]+\b', line): return True
    return False

def looks_like_sentence(line):
    words = line.split()
    if not (3 <= len(words) <= 50): return False
    ascii_words = [w for w in words if re.match(r'^[a-zA-Z.,!?\'"-]+$', w)]
    if len(ascii_words) / (len(words) + 1e-9) > 0.6: return True
    if '"' in line or "'" in line: return True
    return False

# Processing 
print("Parsing OCR publications with 10-step Anchoring...")
for idx, row in tqdm(publications.iterrows(), total=len(publications), desc="OCR Matches"):
    text = str(row['page_text'])
    lines = [clean_text(x) for x in text.split('\n') if x.strip()]
    
    text_lower = text.lower()
    page_aliases_found = False
    for al in alias_map.keys():
        if al.lower() in text_lower:
            page_aliases_found = True
            break
            
    akk_candidates = [(i, l) for i, l in enumerate(lines) if is_akkadian_like(l) and len(l.split()) <= 20]
    trans_candidates = [(i, l) for i, l in enumerate(lines) if looks_like_sentence(l)]
    
    for ai, a_text in akk_candidates:
        score = 0
        if page_aliases_found: score += 5
        
        text_match = False
        a_test = a_text.replace('-', '')
        for ref in reference_lines:
            if a_text in ref or a_test in ref.replace('-',''):
                text_match = True
                break
        if text_match: score += 3
        
        best_dist = 4
        best_eng = None
        for ei, e_text in trans_candidates:
            dist = abs(ai - ei)
            if 0 < dist <= 3 and dist < best_dist:
                best_dist = dist
                best_eng = e_text
                
        if best_eng:
            score += 2
            try:
                en_text, lang = translate_to_english(best_eng)
                if lang == 'en':
                    score += 2
                else:
                    if detect(en_text) == 'en':
                        best_eng = en_text
                        score += 2
            except: pass
            
            if 5 <= len(a_text) <= 500 and 5 <= len(best_eng) <= 500:
                score += 1

            if score >= 7:
                extracted_pairs.append((a_text, best_eng))

final_pairs = list(set(extracted_pairs))

print(f"\nExtraction Complete! Found {len(final_pairs)} pairs.")

with open(os.path.join(train_folder, "try2_read.md"), "w", encoding='utf-8') as f:
    f.write("# Try 2 OCR Extraction Log\n\n")
    f.write(f"- Total extracted pairs: {len(final_pairs)}\n")
    f.write("- Sample pairs:\n")
    for s, t in final_pairs[:5]:
        f.write(f"  - SRC: {s}\n  - TGT: {t}\n")

pd.DataFrame(final_pairs, columns=["Akkadian", "English"]).to_csv(os.path.join(train_folder, "try2_extracted.csv"), index=False)
