#!/usr/bin/env python3
"""
Unified Transliteration Module

Provides consistent, comprehensive transliteration for Bengali and Arabic scripts
using unified mapping tables. Fixes issues with character leakage and inconsistent
mappings across different transliteration schemes.

Usage:
    from scripts.unified_transliteration import transliterate_bengali, transliterate_arabic
    
    bengali_latin = transliterate_bengali("বাংলা")
    arabic_latin = transliterate_arabic("عربي")

Features:
- Proper Nukta (়) handling for Bengali
- Consistent mapping from unified JSON tables
- Context-aware Virama/Hasant processing
- No character leakage into Latin output
- Support for multiple scripts (Bengali, Arabic)

Author: AVEC Project
Date: December 1, 2025
Version: 1.0
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, Optional

class UnifiedTransliterator:
    """
    Unified transliterator using comprehensive mapping tables.
    
    Handles special cases:
    - Bengali Nukta modifier (়)
    - Virama/Hasant (্)
    - Consonant clusters
    - Inherent vowels
    """
    
    def __init__(self, script: str = 'bengali'):
        """
        Initialize transliterator for specified script.
        
        Args:
            script: Language script ('bengali' or 'arabic')
        """
        self.script = script.lower()
        if self.script not in ['bengali', 'arabic']:
            raise ValueError(f"Unsupported script: {script}. Use 'bengali' or 'arabic'.")
        
        self.mapping = self._load_mapping()
        
        # Bengali-specific characters
        if self.script == 'bengali':
            self.nukta = '\u09BC'  # Bengali Sign Nukta
            self.virama = '\u09CD'  # Bengali Sign Virama (Hasant)
            
            # Nukta combinations (base + modifier → composed form)
            self.nukta_combos = {
                'ড' + self.nukta: 'ড়',  # U+09A1 + U+09BC → U+09DC (d → r)
                'ঢ' + self.nukta: 'ঢ়',  # U+09A2 + U+09BC → U+09DD (dh → rh)
                'য' + self.nukta: 'য়',  # U+09AF + U+09BC → U+09DF (j → y)
            }
        
        # Arabic-specific characters
        elif self.script == 'arabic':
            # Arabic diacritics to strip
            self.arabic_diacritics = [
                '\u064B',  # FATHATAN
                '\u064C',  # DAMMATAN
                '\u064D',  # KASRATAN
                '\u064E',  # FATHA
                '\u064F',  # DAMMA
                '\u0650',  # KASRA
                '\u0651',  # SHADDA
                '\u0652',  # SUKUN
                '\u0653',  # MADDAH
                '\u0654',  # HAMZA ABOVE
                '\u0655',  # HAMZA BELOW
                '\u0670',  # SUPERSCRIPT ALEF
                '\u0640',  # TATWEEL (Kashida)
            ]
    
    def _load_mapping(self) -> Dict[str, str]:
        """Load unified mapping table for the specified script."""
        base_dir = Path(__file__).parent.parent
        
        if self.script == 'bengali':
            mapping_file = base_dir / 'transliteration_mappings' / 'unified_bengali_to_latin.json'
        elif self.script == 'arabic':
            # Use existing arabish mapping or create unified one
            mapping_file = base_dir / 'transliteration_mappings' / 'unified_arabic_to_latin.json'
            
            # Fallback to arabish if unified doesn't exist yet
            if not mapping_file.exists():
                return self._get_default_arabic_mapping()
        
        if not mapping_file.exists():
            raise FileNotFoundError(
                f"Mapping file not found: {mapping_file}\n"
                f"Please run analyze_transliteration_coverage.py first."
            )
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_default_arabic_mapping(self) -> Dict[str, str]:
        """Default Arabic mapping (Arabish style) if unified file doesn't exist."""
        return {
            "ا": "a", "أ": "a", "إ": "i", "آ": "aa", "ٱ": "a",
            "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "7", "خ": "kh",
            "د": "d", "ذ": "dh", "ر": "r", "ز": "z", "س": "s", "ش": "sh",
            "ص": "s", "ض": "d", "ط": "t", "ظ": "z",
            "ع": "3", "غ": "gh", "ف": "f", "ق": "q", "ك": "k", "ل": "l",
            "م": "m", "ن": "n", "ه": "h", "و": "w", "ؤ": "w",
            "ي": "y", "ى": "a", "ئ": "y", "ة": "a", "ء": "'"
        }
    
    def transliterate(self, text: str, remove_spaces: bool = False) -> str:
        """
        Transliterate text using unified mapping.
        
        Args:
            text: Input text in source script
            remove_spaces: If True, remove spaces from output
        
        Returns:
            Transliterated text in Latin script
        """
        if not text or not text.strip():
            return text
        
        # Script-specific preprocessing
        if self.script == 'bengali':
            text = self._preprocess_bengali(text)
        elif self.script == 'arabic':
            text = self._preprocess_arabic(text)
        
        # Character-by-character transliteration
        result = []
        for char in text:
            if char in self.mapping:
                mapped = self.mapping[char]
                if mapped:  # Skip empty mappings (like Virama)
                    result.append(mapped)
            elif char.isspace():
                if not remove_spaces:
                    result.append(char)
            elif char in '.,!?-_\'\"':
                result.append(char)
            elif ord(char) < 128:  # Keep ASCII characters
                result.append(char)
            # Skip unmapped non-ASCII characters
        
        # Post-processing
        output = ''.join(result)
        output = self._postprocess(output)
        
        return output
    
    def _preprocess_bengali(self, text: str) -> str:
        """
        Preprocess Bengali text before transliteration.
        
        Handles:
        - Nukta modifier normalization
        - Common ligature decomposition
        """
        # Normalize Nukta: convert base+nukta sequences to precomposed forms
        for decomposed, composed in self.nukta_combos.items():
            text = text.replace(decomposed, composed)
        
        # Remove any remaining stray nuktas
        text = text.replace(self.nukta, '')
        
        # Common ligatures (optional - can add more)
        # ক্ষ = ক + ্ + ষ (ksh)
        # These are usually handled correctly by the base algorithm
        
        return text
    
    def _preprocess_arabic(self, text: str) -> str:
        """
        Preprocess Arabic text before transliteration.
        
        Handles:
        - Diacritic removal
        - Normalization of Alif variants
        """
        # Strip diacritics
        for diacritic in self.arabic_diacritics:
            text = text.replace(diacritic, '')
        
        # Normalize some variants
        # (Most normalization is handled in the mapping table)
        
        return text
    
    def _postprocess(self, text: str) -> str:
        """
        Post-process transliterated text.
        
        - Clean up multiple spaces
        - Remove redundant characters
        - Handle double vowels/consonants
        """
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing spaces
        text = text.strip()
        
        # Optionally handle double consonants (can be customized)
        # For now, keep them as-is
        
        return text
    
    def transliterate_word_list(self, words: list[str]) -> list[str]:
        """
        Transliterate a list of words.
        
        Args:
            words: List of words in source script
        
        Returns:
            List of transliterated words
        """
        return [self.transliterate(word) for word in words]
    
    def validate_output(self, transliterated: str, script_range: tuple[int, int]) -> bool:
        """
        Validate that transliteration contains no source script characters.
        
        Args:
            transliterated: Transliterated text
            script_range: Unicode range tuple (start, end) for source script
        
        Returns:
            True if valid (no leakage), False otherwise
        """
        for char in transliterated:
            if script_range[0] <= ord(char) <= script_range[1]:
                return False
        return True


# Convenience functions

def transliterate_bengali(text: str, remove_spaces: bool = False) -> str:
    """
    Transliterate Bengali text to Latin script.
    
    Args:
        text: Bengali text
        remove_spaces: If True, remove spaces from output
    
    Returns:
        Transliterated text in Latin script
    
    Examples:
        >>> transliterate_bengali("বাংলা")
        'bangla'
        >>> transliterate_bengali("গাড়ি")
        'gari'
        >>> transliterate_bengali("অধ্যয়ন")
        'odhyoyon'
    """
    t = UnifiedTransliterator('bengali')
    return t.transliterate(text, remove_spaces=remove_spaces)


def transliterate_arabic(text: str, remove_spaces: bool = False) -> str:
    """
    Transliterate Arabic text to Latin script.
    
    Args:
        text: Arabic text
        remove_spaces: If True, remove spaces from output
    
    Returns:
        Transliterated text in Latin script (Arabish style)
    
    Examples:
        >>> transliterate_arabic("كتاب")
        'ktab'
        >>> transliterate_arabic("لغة")
        'lgha'
    """
    t = UnifiedTransliterator('arabic')
    return t.transliterate(text, remove_spaces=remove_spaces)


def to_banglish(text: str) -> str:
    """
    Backward-compatible wrapper for Bengali transliteration.
    
    Replaces the old banglish_lookup.to_banglish() function.
    """
    return transliterate_bengali(text, remove_spaces=False)


def to_arabish(text: str) -> str:
    """
    Backward-compatible wrapper for Arabic transliteration.
    
    Replaces the old arabish_lookup.to_arabish() function.
    """
    return transliterate_arabic(text, remove_spaces=False)


# CLI interface
def main():
    """Command-line interface for transliteration."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified transliteration for Bengali and Arabic scripts"
    )
    parser.add_argument('text', nargs='?', help='Text to transliterate')
    parser.add_argument('--script', '-s', choices=['bengali', 'arabic'], 
                       default='bengali', help='Source script')
    parser.add_argument('--file', '-f', help='Input file (one word per line)')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate output (check for character leakage)')
    
    args = parser.parse_args()
    
    trans = UnifiedTransliterator(args.script)
    
    # Determine source
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
    elif args.text:
        words = [args.text]
    else:
        # Interactive mode
        import sys
        print(f"Transliterating from {args.script} to Latin (Ctrl+D to exit):")
        words = [line.strip() for line in sys.stdin if line.strip()]
    
    # Transliterate
    results = []
    for word in words:
        result = trans.transliterate(word)
        results.append(result)
        
        # Validate if requested
        if args.validate:
            if args.script == 'bengali':
                is_valid = trans.validate_output(result, (0x0980, 0x09FF))
            elif args.script == 'arabic':
                is_valid = trans.validate_output(result, (0x0600, 0x06FF))
            
            if not is_valid:
                print(f"WARNING: Character leakage detected in '{word}' → '{result}'",
                      file=sys.stderr)
    
    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(result + '\n')
        print(f"Wrote {len(results)} transliterations to {args.output}")
    else:
        for word, result in zip(words, results):
            if args.file or len(words) > 1:
                print(f"{word} → {result}")
            else:
                print(result)


if __name__ == '__main__':
    main()







