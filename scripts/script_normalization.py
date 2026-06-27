#!/usr/bin/env python3
"""
Script Normalization Utilities for Ablation Study

Provides different script normalization strategies for cross-script transfer learning:
1. raw - Original non-Latin script (Bengali/Arabic)
2. phonetic - IPA-based phonetic romanization  
3. simple - Simple character-level transliteration (e.g., Arabish)
4. mixed - Preserve original + romanization auxiliary

Usage:
    from scripts.script_normalization import normalize_script
    
    # Phonetic romanization
    normalized = normalize_script("কথা", "bengali", mode="phonetic")
    
    # Simple transliteration
    normalized = normalize_script("كتاب", "arabic", mode="simple")
"""

from __future__ import annotations
import re
from typing import Optional

# Import existing arabish utility
try:
    from scripts.arabish_lookup import to_arabish
    ARABISH_AVAILABLE = True
except ImportError:
    ARABISH_AVAILABLE = False


class ScriptNormalizer:
    """Unified script normalization for Bengali and Arabic"""
    
    def __init__(self, script: str = "bengali"):
        """
        Initialize normalizer
        
        Args:
            script: Language script ('bengali' or 'arabic')
        """
        self.script = script.lower()
        assert self.script in ['bengali', 'arabic'], f"Unsupported script: {script}"
    
    def normalize(self, text: str, mode: str = "phonetic") -> str:
        """
        Normalize text according to specified mode
        
        Args:
            text: Input text in original script
            mode: Normalization mode ('raw', 'phonetic', 'simple', 'mixed')
            
        Returns:
            Normalized text
        """
        if not text or not text.strip():
            return text
        
        mode = mode.lower()
        
        if mode == "raw":
            # Keep original script unchanged
            return text
        
        elif mode == "phonetic":
            # IPA-based phonetic romanization
            if self.script == "bengali":
                return self._bengali_phonetic(text)
            else:
                return self._arabic_phonetic(text)
        
        elif mode == "simple":
            # Simple character-level transliteration
            if self.script == "bengali":
                return self._bengali_simple(text)
            else:
                return self._arabic_simple(text)
        
        elif mode == "mixed":
            # Preserve both original and romanization
            if self.script == "bengali":
                romanized = self._bengali_phonetic(text)
            else:
                romanized = self._arabic_phonetic(text)
            return f"{text}|{romanized}"
        
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
    
    def _bengali_phonetic(self, text: str) -> str:
        """
        Bengali to phonetic romanization (IPA-inspired)
        
        This is the approach used in your LipBengal work (Banglish).
        Maps Bengali characters to closest phonetic Latin equivalents.
        """
        # Extended Bengali consonant mapping (phonetic)
        bengali_consonants = {
            'ক': 'k', 'খ': 'kh', 'গ': 'g', 'ঘ': 'gh', 'ঙ': 'ng',
            'চ': 'ch', 'ছ': 'chh', 'জ': 'j', 'ঝ': 'jh', 'ঞ': 'n',
            'ট': 't', 'ঠ': 'th', 'ড': 'd', 'ঢ': 'dh', 'ণ': 'n',
            'ত': 't', 'থ': 'th', 'দ': 'd', 'ধ': 'dh', 'ন': 'n',
            'প': 'p', 'ফ': 'ph', 'ব': 'b', 'ভ': 'bh', 'ম': 'm',
            'য': 'j', 'র': 'r', 'ল': 'l', 'শ': 'sh', 'ষ': 'sh',
            'স': 's', 'হ': 'h', 'ড়': 'r', 'ঢ়': 'rh', 'য়': 'y',
            'ৎ': 't', 'ং': 'ng', 'ঃ': 'h', 'ঁ': 'n'
        }
        
        # Bengali vowels (independent and dependent forms)
        bengali_vowels = {
            'অ': 'o', 'আ': 'a', 'ই': 'i', 'ঈ': 'ee', 'উ': 'u', 'ঊ': 'oo',
            'ঋ': 'ri', 'এ': 'e', 'ঐ': 'oi', 'ও': 'o', 'ঔ': 'ou',
            'া': 'a', 'ি': 'i', 'ী': 'ee', 'ু': 'u', 'ূ': 'oo',
            'ৃ': 'ri', 'ে': 'e', 'ৈ': 'oi', 'ো': 'o', 'ৌ': 'ou'
        }
        
        # Combine mappings
        char_map = {**bengali_consonants, **bengali_vowels}
        
        result = []
        for char in text:
            if char in char_map:
                result.append(char_map[char])
            elif char.isspace() or char in '.,!?-_':
                result.append(char)
            else:
                # Keep ASCII characters as-is
                if ord(char) < 128:
                    result.append(char)
                else:
                    result.append('x')  # Unknown Bengali char
        
        romanized = ''.join(result)
        # Clean up
        romanized = re.sub(r'x+', '', romanized)
        romanized = re.sub(r'\s+', ' ', romanized)
        return romanized.strip()
    
    def _bengali_simple(self, text: str) -> str:
        """
        Simple Bengali transliteration (more literal, less phonetic)
        Maps each Bengali character to a single Latin character where possible.
        """
        # Simplified 1:1 mapping
        simple_map = {
            'ক': 'k', 'খ': 'K', 'গ': 'g', 'ঘ': 'G', 'ঙ': 'N',
            'চ': 'c', 'ছ': 'C', 'জ': 'j', 'ঝ': 'J', 'ঞ': 'n',
            'ট': 'T', 'ঠ': 't', 'ড': 'D', 'ঢ': 'd', 'ণ': 'n',
            'ত': 't', 'থ': 'T', 'দ': 'd', 'ধ': 'D', 'ন': 'n',
            'প': 'p', 'ফ': 'f', 'ব': 'b', 'ভ': 'v', 'ম': 'm',
            'য': 'z', 'র': 'r', 'ল': 'l', 'শ': 's', 'ষ': 'S',
            'স': 's', 'হ': 'h', 'ড়': 'r', 'ঢ়': 'R', 'য়': 'y',
            'অ': 'a', 'আ': 'A', 'ই': 'i', 'ঈ': 'I', 'উ': 'u', 'ঊ': 'U',
            'ঋ': 'R', 'এ': 'e', 'ঐ': 'E', 'ও': 'o', 'ঔ': 'O',
            'া': 'a', 'ি': 'i', 'ী': 'I', 'ু': 'u', 'ূ': 'U',
            'ৃ': 'R', 'ে': 'e', 'ৈ': 'E', 'ো': 'o', 'ৌ': 'O'
        }
        
        result = []
        for char in text:
            if char in simple_map:
                result.append(simple_map[char])
            elif ord(char) < 128:
                result.append(char)
        
        return ''.join(result).strip()
    
    def _arabic_phonetic(self, text: str) -> str:
        """
        Arabic to phonetic romanization (IPA-inspired)
        
        Uses more phonetically accurate representations than simple Arabish.
        """
        # Phonetic Arabic mapping
        phonetic_map = {
            "ا": "a", "أ": "a", "إ": "i", "آ": "aa", "ٱ": "a",
            "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "h", "خ": "kh",
            "د": "d", "ذ": "dh", "ر": "r", "ز": "z", "س": "s", "ش": "sh",
            "ص": "s", "ض": "d", "ط": "t", "ظ": "z",
            "ع": "'", "غ": "gh", "ف": "f", "ق": "q", "ك": "k", "ل": "l",
            "م": "m", "ن": "n", "ه": "h", "و": "w", "ؤ": "w",
            "ي": "y", "ى": "a", "ئ": "y", "ة": "h", "ء": "'"
        }
        
        # Strip diacritics
        text = self._strip_arabic_diacritics(text)
        
        result = []
        for char in text:
            if char in phonetic_map:
                result.append(phonetic_map[char])
            elif ord(char) < 128:
                result.append(char)
        
        romanized = ''.join(result)
        romanized = re.sub(r"'+", "", romanized)  # Remove glottal stops in simplified
        romanized = re.sub(r'\s+', ' ', romanized)
        return romanized.strip()
    
    def _arabic_simple(self, text: str) -> str:
        """
        Simple Arabic transliteration (Arabish style)
        Uses the existing arabish_lookup if available.
        """
        if ARABISH_AVAILABLE:
            return to_arabish(text)
        else:
            # Fallback to phonetic if arabish not available
            return self._arabic_phonetic(text)
    
    def _strip_arabic_diacritics(self, text: str) -> str:
        """Remove Arabic diacritical marks"""
        diacritics = [
            '\u064b',  # FATHATAN
            '\u064c',  # DAMMATAN
            '\u064d',  # KASRATAN
            '\u064e',  # FATHA
            '\u064f',  # DAMMA
            '\u0650',  # KASRA
            '\u0651',  # SHADDA
            '\u0652',  # SUKUN
            '\u0653',  # MADDAH
            '\u0654',  # HAMZA ABOVE
            '\u0655',  # HAMZA BELOW
            '\u0670',  # SUPERSCRIPT ALEF
            '\u0640',  # TATWEEL
        ]
        for d in diacritics:
            text = text.replace(d, '')
        return text


def normalize_script(text: str, script: str, mode: str = "phonetic") -> str:
    """
    Convenience function for script normalization
    
    Args:
        text: Input text
        script: Script name ('bengali' or 'arabic')
        mode: Normalization mode ('raw', 'phonetic', 'simple', 'mixed')
        
    Returns:
        Normalized text
    """
    normalizer = ScriptNormalizer(script)
    return normalizer.normalize(text, mode)


# Backward compatibility wrappers
def to_banglish(text: str, mode: str = "phonetic") -> str:
    """Convert Bengali to Banglish (romanized Bengali)"""
    return normalize_script(text, "bengali", mode=mode)


def to_arabish_phonetic(text: str) -> str:
    """Convert Arabic to phonetic romanization"""
    return normalize_script(text, "arabic", mode="phonetic")


def main():
    """Test script normalization"""
    import sys
    
    # Test Bengali
    bengali_tests = [
        "কথা",  # kotha
        "আমি",  # ami
        "বাংলা",  # bangla
    ]
    
    # Test Arabic  
    arabic_tests = [
        "كتاب",  # kitab
        "لغة",   # lugha
        "مدرسة", # madrasa
    ]
    
    print("Bengali Normalization Tests:")
    print("=" * 60)
    for text in bengali_tests:
        print(f"Original: {text}")
        print(f"  Raw:      {normalize_script(text, 'bengali', 'raw')}")
        print(f"  Phonetic: {normalize_script(text, 'bengali', 'phonetic')}")
        print(f"  Simple:   {normalize_script(text, 'bengali', 'simple')}")
        print(f"  Mixed:    {normalize_script(text, 'bengali', 'mixed')}")
        print()
    
    print("\nArabic Normalization Tests:")
    print("=" * 60)
    for text in arabic_tests:
        print(f"Original: {text}")
        print(f"  Raw:      {normalize_script(text, 'arabic', 'raw')}")
        print(f"  Phonetic: {normalize_script(text, 'arabic', 'phonetic')}")
        print(f"  Simple:   {normalize_script(text, 'arabic', 'simple')}")
        print(f"  Mixed:    {normalize_script(text, 'arabic', 'mixed')}")
        print()


if __name__ == "__main__":
    main()


