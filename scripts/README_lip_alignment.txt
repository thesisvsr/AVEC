================================================================================
LIP ALIGNMENT VISUALIZATION - QUICK REFERENCE
================================================================================

MAIN SCRIPT:
  scripts/create_lip_alignment_enhanced.py

USAGE EXAMPLES:
  # Random word from speaker s100
  python scripts/create_lip_alignment_enhanced.py --speaker s100

  # Specific word
  python scripts/create_lip_alignment_enhanced.py --speaker s100 --word "আকাশ"

  # Custom output
  python scripts/create_lip_alignment_enhanced.py \
      --speaker s100 \
      --word "ছোট ভাই" \
      --output export/my_viz.png \
      --max-frames-per-row 8

VISUALIZATION SHOWS:
  1. Bengali word (top) with proper font
  2. Arrow pointing down
  3. English transliteration (Banglish)
  4. All preprocessed lip frames in a grid
  5. Individual English characters above corresponding frames
  6. Frame numbers below each frame

OUTPUT FILES:
  - {output}.png  : The visualization image
  - {output}.json : Metadata (word, characters, alignment, etc.)

EXAMPLES CREATED:
  1. export/paper_samples/lip_alignment_enhanced.png   (ছোট ভাই - choto bhai)
  2. export/paper_samples/lip_alignment_akash.png      (আকাশ - akasho)
  3. export/paper_samples/lip_alignment_adhyoyon.png   (অধ্যয়ন - odhyoyono)

REQUIREMENTS:
  - Activate .venv: source .venv/bin/activate
  - Bengali font installed (Noto Sans Bengali recommended)
  - Preprocessed data available in datasets/LipBengal/prepared/

================================================================================
