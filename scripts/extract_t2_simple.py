#!/usr/bin/env python3
"""Extract T2 results without needing TensorBoard"""

import struct
import os
from pathlib import Path

def read_tfrecord(path):
    """Simple TFRecord reader - reads raw events"""
    with open(path, 'rb') as f:
        while True:
            # Read length
            length_bytes = f.read(8)
            if len(length_bytes) < 8:
                break
            
            length = struct.unpack('<Q', length_bytes)[0]
            if length > 1000000:  # Skip invalid records
                break
                
            # Read CRC
            f.read(4)
            
            # Read data
            data = f.read(length)
            if len(data) < length:
                break
            
            # Read CRC
            f.read(4)
            
            # Try to decode metrics from raw data
            data_str = data.decode('latin1', errors='ignore')
            
            # Look for evaluation metrics
            if 'Evaluation-epoch' in data_str:
                if '/acc' in data_str:
                    yield ('acc', data)
                elif '/wer' in data_str:
                    yield ('wer', data)
                elif '/cer' in data_str:
                    yield ('cer', data)

def find_event_files(log_dir):
    """Find all event files in directory"""
    return list(Path(log_dir).glob('events.out.tfevents.*'))

def extract_best_metrics(log_dir):
    """Extract best metrics from event files"""
    event_files = find_event_files(log_dir)
    if not event_files:
        print(f"No event files found in {log_dir}")
        return None
    
    # Use the most recent event file
    latest_event = max(event_files, key=lambda p: p.stat().st_mtime)
    print(f"Reading: {latest_event.name}")
    
    acc_values = []
    wer_values = []
    cer_values = []
    
    try:
        for metric_type, data in read_tfrecord(str(latest_event)):
            # Extract float values from binary data
            # This is a simplified approach - looking for float patterns
            for i in range(len(data) - 4):
                try:
                    value = struct.unpack('<f', data[i:i+4])[0]
                    # Filter reasonable values (0-100 for percentages)
                    if 0 <= value <= 100:
                        if metric_type == 'acc':
                            acc_values.append(value)
                        elif metric_type == 'wer':
                            wer_values.append(value)
                        elif metric_type == 'cer':
                            cer_values.append(value)
                except:
                    continue
    except Exception as e:
        print(f"Error reading events: {e}")
    
    if acc_values:
        return {
            'accuracy': max(acc_values),
            'wer': min(wer_values) if wer_values else None,
            'cer': min(cer_values) if cer_values else None
        }
    return None

def main():
    experiments = {
        'T2_freeze_0ep': 'callbacks/LipBengal/AV/ablations/T2_freeze_0ep/logs',
        'T2_freeze_10ep': 'callbacks/LipBengal/AV/ablations/T2_freeze_10ep/logs',
    }
    
    print("=" * 80)
    print("Extracting T2 Results (Simple Method)")
    print("=" * 80)
    print()
    
    results = {}
    for exp_name, log_dir in experiments.items():
        if not Path(log_dir).exists():
            print(f"⚠️  {exp_name}: Directory not found")
            continue
        
        print(f"Processing: {exp_name}")
        metrics = extract_best_metrics(log_dir)
        
        if metrics:
            results[exp_name] = metrics
            print(f"  ✓ Accuracy: {metrics['accuracy']:.2f}%")
            if metrics['wer']:
                print(f"  ✓ WER: {metrics['wer']:.2f}%")
            if metrics['cer']:
                print(f"  ✓ CER: {metrics['cer']:.2f}%")
        else:
            print(f"  ✗ Could not extract metrics")
        print()
    
    # Print CSV format
    if results:
        print("=" * 80)
        print("CSV Format:")
        print("=" * 80)
        print("Experiment,Accuracy,WER,CER")
        for exp_name, metrics in results.items():
            wer = metrics['wer'] if metrics['wer'] else 'N/A'
            cer = metrics['cer'] if metrics['cer'] else 'N/A'
            print(f"{exp_name},{metrics['accuracy']},{wer},{cer}")

if __name__ == '__main__':
    main()




