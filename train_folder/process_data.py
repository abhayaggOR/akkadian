import pandas as pd
import re
import os
import random

cwd = "/Users/abhayaggarwal/Downloads/deep-past-initiative-machine-translation"
train_folder = os.path.join(cwd, "train_folder")
os.makedirs(train_folder, exist_ok=True)

md_file = open(os.path.join(train_folder, "read.md"), "w", encoding='utf-8')
md_file.write("# Detailed Processing Log\n\n")

print("Starting Phase 1...")
# Phase 1
md_file.write("## Phase 1: Convert train.csv into sentence-level parallel data\n")
train_df = pd.read_csv(os.path.join(cwd, "train.csv"))
source_sentences = []
target_sentences = []

for _, row in train_df.iterrows():
    if pd.isna(row['transliteration']) or pd.isna(row['translation']): continue
    t_lines = [x.strip() for x in re.split(r'[\n\.]', str(row['transliteration'])) if x.strip()]
    e_lines = [x.strip() for x in re.split(r'[\n\.]', str(row['translation'])) if x.strip()]
    
    for s, t in zip(t_lines, e_lines):
        if len(s) >= 5 and len(t) >= 5:
            source_sentences.append(s)
            target_sentences.append(t)

md_file.write(f"- Num pairs generated: {len(source_sentences)}\n")
md_file.write("- Sample pairs:\n")
for s, t in zip(source_sentences[:5], target_sentences[:5]):
    md_file.write(f"  - SRC: {s}\n  - TGT: {t}\n")

print("Starting Phase 2...")
# Phase 2
md_file.write("\n## Phase 2: Clean sentence pairs\n")
def clean_text(text):
    return re.sub(r'\s+', ' ', text.lower()).strip()

def is_only_punct_or_nums(text):
    return not bool(re.search(r'[a-zšṣṭḫāīū]', text, re.I))

c_source = []
c_target = []
seen = set()

for s, t in zip(source_sentences, target_sentences):
    s_c = clean_text(s)
    t_c = clean_text(t)
    if is_only_punct_or_nums(s_c) or is_only_punct_or_nums(t_c): continue
    if (s_c, t_c) not in seen:
        seen.add((s_c, t_c))
        c_source.append(s_c)
        c_target.append(t_c)

md_file.write(f"- Pairs after cleaning: {len(c_source)}\n")
md_file.write("- 5 Sample pairs:\n")
for s, t in zip(c_source[:5], c_target[:5]):
    md_file.write(f"  - SRC: {s}\n  - TGT: {t}\n")

print("Starting Phase 3...")
# Phase 3
md_file.write("\n## Phase 3: Create training files (train.src, train.tgt)\n")
with open(os.path.join(train_folder, "train.src"), "w", encoding='utf-8') as f:
    f.write("\n".join(c_source) + "\n")
with open(os.path.join(train_folder, "train.tgt"), "w", encoding='utf-8') as f:
    f.write("\n".join(c_target) + "\n")
md_file.write(f"- Lines written: {len(c_source)}\n")
md_file.write("- First 5 lines (train.src):\n")
for l in c_source[:5]: md_file.write(f"  - {l}\n")
md_file.write("- First 5 lines (train.tgt):\n")
for l in c_target[:5]: md_file.write(f"  - {l}\n")

print("Starting Phase 4...")
# Phase 4
md_file.write("\n## Phase 4: Create single corpus file for tokenizer from published_texts.csv\n")
pub_texts = pd.read_csv(os.path.join(cwd, "published_texts.csv"))
akk_corpus = list(c_source)
for t in pub_texts['transliteration'].dropna():
    lines = [x.strip() for x in re.split(r'[\n\.]', str(t)) if x.strip()]
    for l in lines:
        l_c = clean_text(l)
        if len(l_c) >= 5 and not is_only_punct_or_nums(l_c):
            akk_corpus.append(l_c)

with open(os.path.join(train_folder, "corpus.src"), "w", encoding='utf-8') as f:
    f.write("\n".join(akk_corpus) + "\n")
md_file.write(f"- Total lines in corpus: {len(akk_corpus)}\n")
md_file.write("- Sample 5 lines:\n")
for l in akk_corpus[-5:]: md_file.write(f"  - {l}\n")

print("Starting Phase 5...")
# Phase 5
md_file.write("\n## Phase 5: Extract Akkadian-English pairs from publications.csv\n")
pubs = pd.read_csv(os.path.join(cwd, "publications.csv"))
if 'has_akkadian' in pubs.columns:
    pubs = pubs[pubs['has_akkadian'] == True]

extracted_pairs = []

def is_akkadian_like(line):
    if re.search(r'[šṣṭḫāīū]', line): return True
    if re.search(r'\b[a-z]+-[a-z]+\b', line): return True
    return False

def is_english_like(line):
    words = line.split()
    if not words: return False
    ascii_words = [w for w in words if re.match(r'^[a-z.,!?:]+$', w)]
    if len(ascii_words) / (len(words) + 1e-9) > 0.7:
        common = {'the', 'and', 'to', 'of', 'in', 'i', 'that', 'you', 'it', 'for'}
        if any(w in common for w in ascii_words): return True
    return False

for _, row in pubs.iterrows():
    text = str(row['page_text'])
    lines = [clean_text(x) for x in text.split('\n') if x.strip()]
    
    akk_idx = [(i, l) for i, l in enumerate(lines) if is_akkadian_like(l)]
    eng_idx = [(i, l) for i, l in enumerate(lines) if is_english_like(l)]
    
    matched_eng = set()
    for ai, a_text in akk_idx:
        best_eng = None
        best_dist = 4
        for ei, e_text in eng_idx:
            if ei in matched_eng: continue
            dist = abs(ai - ei)
            if dist > 0 and dist < best_dist:
                best_dist = dist
                best_eng = (ei, e_text)
        if best_eng:
            matched_eng.add(best_eng[0])
            extracted_pairs.append((a_text, best_eng[1]))

md_file.write(f"- Number of extracted pairs: {len(extracted_pairs)}\n")
md_file.write("- 5 Examples:\n")
for s, t in extracted_pairs[:5]:
    md_file.write(f"  - SRC: {s}\n  - TGT: {t}\n")

print("Starting Phase 6...")
# Phase 6
md_file.write("\n## Phase 6: Combine datasets\n")
all_pairs = list(zip(c_source, c_target))

extracted_clean = []
for a, e in extracted_pairs:
    if len(a) >= 5 and len(e) >= 5:
        if not is_only_punct_or_nums(a) and not is_only_punct_or_nums(e):
            extracted_clean.append((a, e))

all_pairs.extend(extracted_clean)
unique_pairs = list(set(all_pairs))
random.seed(42)
random.shuffle(unique_pairs)

md_file.write(f"- Final number of sentence pairs: {len(unique_pairs)}\n")
md_file.write("- 5 Sample pairs:\n")
for s, t in unique_pairs[:5]: md_file.write(f"  - SRC: {s}\n  - TGT: {t}\n")
perc_orig = (len(c_source) / len(unique_pairs)) * 100 if unique_pairs else 0
perc_ext = (len(extracted_clean) / len(unique_pairs)) * 100 if unique_pairs else 0
md_file.write(f"- % from original: {perc_orig:.2f}%\n")
md_file.write(f"- % from extracted: {perc_ext:.2f}%\n")

print("Starting Phase 7...")
# Phase 7
md_file.write("\n## Phase 7: Prepare final dataset\n")
final_src = [p[0] for p in unique_pairs]
final_tgt = [p[1] for p in unique_pairs]
with open(os.path.join(train_folder, "final_train.src"), "w", encoding='utf-8') as f:
    f.write("\n".join(final_src) + "\n")
with open(os.path.join(train_folder, "final_train.tgt"), "w", encoding='utf-8') as f:
    f.write("\n".join(final_tgt) + "\n")

avg_src = sum(len(s) for s in final_src) / len(final_src) if final_src else 0
avg_tgt = sum(len(t) for t in final_tgt) / len(final_tgt) if final_tgt else 0
md_file.write(f"- Total training pairs: {len(unique_pairs)}\n")
md_file.write(f"- Average sentence length (source): {avg_src:.2f} chars\n")
md_file.write(f"- Average sentence length (target): {avg_tgt:.2f} chars\n")

md_file.close()
print("Processing Completed Successfully.")
