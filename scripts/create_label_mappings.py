#!/usr/bin/env python3
"""
Create label mapping files for LipBengal script normalization ablation.

Generates mappings from phonetic romanization (Banglish) to:
- Raw Bengali script
- Simple transliteration
- Mixed approach

Output: datasets/LipBengal/label_mappings.json
"""

import json
from pathlib import Path

def load_word_lists():
    """Load both word lists"""
    bengali_file = Path("datasets/LipBengal/word_list.txt")
    banglish_file = Path("datasets/LipBengal/word_list_banglish.txt")
    
    with open(bengali_file, 'r', encoding='utf-8') as f:
        bengali_words = [line.strip() for line in f if line.strip()]
    
    with open(banglish_file, 'r', encoding='utf-8') as f:
        banglish_words = [line.strip() for line in f if line.strip()]
    
    if len(bengali_words) != len(banglish_words):
        print(f"Warning: Word list lengths don't match: {len(bengali_words)} vs {len(banglish_words)}")
    
    return bengali_words, banglish_words


def create_simple_transliteration(bengali_word):
    """
    Simple character-by-character transliteration
    Maps Bengali characters to closest Latin equivalents
    """
    # Bengali consonants to Latin
    consonant_map = {
        'ক': 'k', 'খ': 'kh', 'গ': 'g', 'ঘ': 'gh', 'ঙ': 'ng',
        'চ': 'ch', 'ছ': 'chh', 'জ': 'j', 'ঝ': 'jh', 'ঞ': 'ny',
        'ট': 't', 'ঠ': 'th', 'ড': 'd', 'ঢ': 'dh', 'ণ': 'n',
        'ত': 't', 'থ': 'th', 'দ': 'd', 'ধ': 'dh', 'ন': 'n',
        'প': 'p', 'ফ': 'ph', 'ব': 'b', 'ভ': 'bh', 'ম': 'm',
        'য': 'j', 'র': 'r', 'ল': 'l', 'শ': 'sh', 'ষ': 'sh',
        'স': 's', 'হ': 'h', 'ড়': 'r', 'ঢ়': 'rh', 'য়': 'y',
        'ৎ': 't', 'ং': 'ng', 'ঃ': 'h', 'ঁ': 'n',
    }
    
    # Bengali vowels to Latin
    vowel_map = {
        'অ': 'a', 'আ': 'aa', 'ই': 'i', 'ঈ': 'ii', 'উ': 'u', 'ঊ': 'uu',
        'ঋ': 'ri', 'এ': 'e', 'ঐ': 'ai', 'ও': 'o', 'ঔ': 'au',
    }
    
    # Vowel diacritics (matra)
    diacritic_map = {
        'া': 'aa', 'ি': 'i', 'ী': 'ii', 'ু': 'u', 'ূ': 'uu',
        'ৃ': 'ri', 'ে': 'e', 'ৈ': 'ai', 'ো': 'o', 'ৌ': 'au',
        '্': '', 'ঁ': 'n', 'ং': 'ng', 'ঃ': 'h',
    }
    
    result = []
    for char in bengali_word:
        if char in consonant_map:
            result.append(consonant_map[char])
        elif char in vowel_map:
            result.append(vowel_map[char])
        elif char in diacritic_map:
            result.append(diacritic_map[char])
        else:
            # Keep unknown characters as-is
            result.append(char)
    
    return ''.join(result)


def create_mixed_approach(phonetic, simple):
    """
    Mixed: Use simpler form but keep some phonetic accuracy
    Take phonetic as base, simplify doubled consonants
    """
    # Simplify doubled consonants
    mixed = phonetic
    mixed = mixed.replace('chh', 'ch')
    mixed = mixed.replace('jjh', 'jh')
    mixed = mixed.replace('kh', 'k')
    mixed = mixed.replace('gh', 'g')
    mixed = mixed.replace('th', 't')
    mixed = mixed.replace('dh', 'd')
    mixed = mixed.replace('bh', 'b')
    mixed = mixed.replace('ph', 'f')
    
    # But keep important distinctions
    # (This is a simplified approach - real mixed would be more sophisticated)
    return mixed


def main():
    print("=" * 70)
    print("Creating LipBengal Label Mappings for Script Normalization")
    print("=" * 70)
    print()
    
    # Load word lists
    print("Loading word lists...")
    bengali_words, banglish_words = load_word_lists()
    print(f"  ✓ Loaded {len(bengali_words)} Bengali words")
    print(f"  ✓ Loaded {len(banglish_words)} Banglish words")
    print()
    
    # Create mappings
    mappings = {
        'phonetic_to_raw': {},
        'phonetic_to_simple': {},
        'phonetic_to_mixed': {},
        'raw_to_phonetic': {},
    }
    
    print("Generating mappings...")
    for bengali, banglish in zip(bengali_words, banglish_words):
        # Phonetic (Banglish) → Raw Bengali
        mappings['phonetic_to_raw'][banglish] = bengali
        mappings['raw_to_phonetic'][bengali] = banglish
        
        # Phonetic → Simple transliteration
        simple = create_simple_transliteration(bengali)
        mappings['phonetic_to_simple'][banglish] = simple
        
        # Phonetic → Mixed approach
        mixed = create_mixed_approach(banglish, simple)
        mappings['phonetic_to_mixed'][banglish] = mixed
    
    print(f"  ✓ Created {len(mappings['phonetic_to_raw'])} mappings")
    print()
    
    # Show examples
    print("Example mappings:")
    examples = banglish_words[:5]
    for phonetic in examples:
        raw = mappings['phonetic_to_raw'].get(phonetic, '?')
        simple = mappings['phonetic_to_simple'].get(phonetic, '?')
        mixed = mappings['phonetic_to_mixed'].get(phonetic, '?')
        print(f"  Phonetic: {phonetic:20s} → Raw: {raw:15s} | Simple: {simple:15s} | Mixed: {mixed:15s}")
    print()
    
    # Save to file
    output_path = Path("datasets/LipBengal/label_mappings.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved mappings to: {output_path}")
    print()
    
    # Generate vocabulary files for each format
    vocab_dir = Path("datasets/LipBengal/vocab")
    vocab_dir.mkdir(exist_ok=True)
    
    formats = {
        'phonetic': banglish_words,
        'raw': bengali_words,
        'simple': [mappings['phonetic_to_simple'][w] for w in banglish_words],
        'mixed': [mappings['phonetic_to_mixed'][w] for w in banglish_words],
    }
    
    for format_name, words in formats.items():
        vocab_file = vocab_dir / f"word_list_{format_name}.txt"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for word in words:
                f.write(f"{word}\n")
        print(f"✓ Saved vocabulary: {vocab_file}")
    
    print()
    print("=" * 70)
    print("✓ Label mappings created successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()


