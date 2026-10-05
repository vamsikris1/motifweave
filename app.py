import os
import sys
import io
import base64
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image
import torch
from flask import Flask, request, jsonify, send_file, render_template_string

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_model, embed_image, get_inference_transform
from src.retrieval import load_gallery, retrieve, verify_pair, FROZEN_VAL_EER_THRESHOLD, FROZEN_VAL_MAX_F1_THRESHOLD
from scripts.create_color_variants import rgb_to_hsv_np, hsv_to_rgb_np, generate_colorway_1, generate_colorway_2, generate_colorway_3

app = Flask(__name__)

# Global model and gallery cache
print("Initializing MotifWeave AI model and gallery index...")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = PROJECT_ROOT / "outputs/checkpoints/baseline_best.pth"
GALLERY_PATH = PROJECT_ROOT / "outputs/final/galleries/test_gallery.npz"
TEST_MANIFEST_PATH = PROJECT_ROOT / "data/processed/splits/test.csv"

model, config, device = load_model(CHECKPOINT_PATH, device=DEVICE)
gallery = load_gallery(GALLERY_PATH)
transform = get_inference_transform(224)

# Load test manifest for sample references
test_df = pd.read_csv(TEST_MANIFEST_PATH) if TEST_MANIFEST_PATH.exists() else pd.DataFrame()
print(f"MotifWeave initialized successfully on {DEVICE} with {gallery['num_items']} gallery items.")

def pil_to_base64(img: Image.Image, format="JPEG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format, quality=90)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_image_from_request(req, file_key='image', path_key='image_path') -> Image.Image:
    if file_key in req.files and req.files[file_key].filename != '':
        file_obj = req.files[file_key]
        return Image.open(file_obj.stream).convert('RGB')
    data = req.get_json(silent=True) or req.form
    if path_key in data and data[path_key]:
        p = Path(data[path_key])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            return Image.open(p).convert('RGB')
        raise FileNotFoundError(f"Image not found at path: {data[path_key]}")
    raise ValueError(f"No image provided for key '{file_key}' or '{path_key}'.")

@app.route('/media/<path:subpath>')
def serve_media(subpath):
    # Resolve relative to PROJECT_ROOT
    target_path = (PROJECT_ROOT / subpath).resolve()
    if not str(target_path).startswith(str(PROJECT_ROOT.resolve())):
        return jsonify({'error': 'Unauthorized path access'}), 403
    if not target_path.exists() or not target_path.is_file():
        return jsonify({'error': 'File not found'}), 404
    return send_file(target_path)

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        'status': 'online',
        'device': str(DEVICE),
        'model_name': 'ConvNeXt-Tiny + GeM + 2-Layer MLP',
        'trainable_params': '526,081 (1.86%)',
        'frozen_params': '27,818,592 (98.14%)',
        'embedding_dim': 256,
        'normalization': 'L2 Unit Hypersphere',
        'gallery_items': gallery['num_items'],
        'thresholds': {
            'val_eer': FROZEN_VAL_EER_THRESHOLD,
            'val_max_f1': FROZEN_VAL_MAX_F1_THRESHOLD
        },
        'benchmarks': {
            'test_recall_at_1': '97.50%',
            'test_recall_at_5': '100.00%',
            'test_map': '98.75%',
            'test_roc_auc': '0.9967',
            'test_separation_margin': '+0.5966'
        }
    })

@app.route('/api/samples', methods=['GET'])
def api_samples():
    if test_df.empty:
        return jsonify([])
    
    samples = []
    # Pick distinct sources
    unique_sources = test_df['source_id'].unique()[:8]
    for s_id in unique_sources:
        sub = test_df[test_df['source_id'] == s_id]
        orig_row = sub[sub['variant_type'] == 'original']
        var1_row = sub[sub['variant_type'] == 'color_variant_1']
        var2_row = sub[sub['variant_type'] == 'color_variant_2']
        var3_row = sub[sub['variant_type'] == 'color_variant_3']
        
        orig_path = orig_row.iloc[0]['image_path'] if len(orig_row) > 0 else None
        var1_path = var1_row.iloc[0]['image_path'] if len(var1_row) > 0 else None
        var2_path = var2_row.iloc[0]['image_path'] if len(var2_row) > 0 else None
        var3_path = var3_row.iloc[0]['image_path'] if len(var3_row) > 0 else None
        
        rel_orig = str(Path(orig_path).relative_to(PROJECT_ROOT)).replace('\\', '/') if orig_path else ''
        rel_var1 = str(Path(var1_path).relative_to(PROJECT_ROOT)).replace('\\', '/') if var1_path else ''
        rel_var2 = str(Path(var2_path).relative_to(PROJECT_ROOT)).replace('\\', '/') if var2_path else ''
        rel_var3 = str(Path(var3_path).relative_to(PROJECT_ROOT)).replace('\\', '/') if var3_path else ''
        
        samples.append({
            'source_id': s_id,
            'original': {'path': rel_orig, 'url': f'/media/{rel_orig}', 'title': f'{s_id} (Original)'},
            'variant1': {'path': rel_var1, 'url': f'/media/{rel_var1}', 'title': f'{s_id} (+120° Triadic)'},
            'variant2': {'path': rel_var2, 'url': f'/media/{rel_var2}', 'title': f'{s_id} (+240° Triadic)'},
            'variant3': {'path': rel_var3, 'url': f'/media/{rel_var3}', 'title': f'{s_id} (+50° Dye Shift)'}
        })
    return jsonify(samples)

@app.route('/api/gallery', methods=['GET'])
def api_gallery():
    items = []
    for idx in range(gallery['num_items']):
        full_path = Path(gallery['image_paths'][idx])
        try:
            rel_path = str(full_path.relative_to(PROJECT_ROOT)).replace('\\', '/')
        except ValueError:
            rel_path = str(full_path).replace('\\', '/')
        items.append({
            'index': idx,
            'source_id': gallery['source_ids'][idx] if idx < len(gallery['source_ids']) else 'unknown',
            'filename': full_path.name,
            'path': rel_path,
            'url': f'/media/{rel_path}'
        })
    return jsonify(items)

@app.route('/api/query', methods=['POST'])
def api_query():
    try:
        top_k = int(request.form.get('top_k', 6))
        # Handle file upload or relative/absolute path
        img = get_image_from_request(request, 'query_image', 'query_path')
        
        # Extract embedding
        query_emb = embed_image(img, model, transform, DEVICE)
        
        # Retrieve matches
        results = retrieve(query_emb, gallery, top_k=top_k, model=model, device=DEVICE)
        
        formatted_results = []
        for r in results:
            full_path = Path(r['image_path'])
            try:
                rel_path = str(full_path.relative_to(PROJECT_ROOT)).replace('\\', '/')
            except ValueError:
                rel_path = str(full_path).replace('\\', '/')
            formatted_results.append({
                'rank': r['rank'],
                'similarity': float(r['similarity']),
                'source_id': r.get('source_id', ''),
                'filename': full_path.name,
                'path': rel_path,
                'url': f'/media/{rel_path}',
                'is_match': float(r['similarity']) >= FROZEN_VAL_EER_THRESHOLD
            })
            
        return jsonify({
            'success': True,
            'top_k': top_k,
            'results': formatted_results,
            'query_embedding_preview': [round(float(v), 4) for v in query_emb[:8].tolist()],
            'embedding_norm': float(np.linalg.norm(query_emb))
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/verify', methods=['POST'])
def api_verify():
    try:
        threshold_mode = request.form.get('threshold_mode', 'eer')
        custom_thresh = request.form.get('threshold', None)
        if custom_thresh:
            threshold = float(custom_thresh)
        elif threshold_mode == 'max_f1':
            threshold = FROZEN_VAL_MAX_F1_THRESHOLD
        else:
            threshold = FROZEN_VAL_EER_THRESHOLD
            
        img_a = get_image_from_request(request, 'image_a', 'image_a_path')
        img_b = get_image_from_request(request, 'image_b', 'image_b_path')
        
        emb_a = embed_image(img_a, model, transform, DEVICE)
        emb_b = embed_image(img_b, model, transform, DEVICE)
        
        similarity = float(np.dot(emb_a, emb_b))
        is_same = similarity >= threshold
        
        return jsonify({
            'success': True,
            'similarity': round(similarity, 4),
            'threshold': round(threshold, 4),
            'threshold_mode': threshold_mode,
            'prediction': 'SAME' if is_same else 'DIFFERENT',
            'is_same': is_same,
            'margin': round(similarity - threshold, 4),
            'eer_decision': 'SAME' if similarity >= FROZEN_VAL_EER_THRESHOLD else 'DIFFERENT',
            'f1_decision': 'SAME' if similarity >= FROZEN_VAL_MAX_F1_THRESHOLD else 'DIFFERENT',
            'norm_a': float(np.linalg.norm(emb_a)),
            'norm_b': float(np.linalg.norm(emb_b)),
            'summary': f"Cosine similarity is {similarity:.4f}. {'Above' if is_same else 'Below'} operating threshold of {threshold:.4f}."
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/simulate_shift', methods=['POST'])
def api_simulate_shift():
    try:
        shift_type = request.form.get('shift_type', 'variant1')
        custom_hue = float(request.form.get('custom_hue', 0.333))
        
        img = get_image_from_request(request, 'image', 'image_path')
        
        if shift_type == 'variant1':
            shifted_img = generate_colorway_1(img)
            shift_label = "+120° Triadic Hue Shift"
        elif shift_type == 'variant2':
            shifted_img = generate_colorway_2(img)
            shift_label = "+240° Triadic Hue Shift"
        elif shift_type == 'variant3':
            shifted_img = generate_colorway_3(img)
            shift_label = "+50° Antique Dye Shift"
        elif shift_type == 'grayscale':
            # RGB grayscale preserving edge textures
            gray = img.convert('L').convert('RGB')
            shifted_img = gray
            shift_label = "Complete Achromatic Grayscale"
        elif shift_type == 'custom':
            arr = np.array(img.convert('RGB'), dtype=np.float32) / 255.0
            hsv = rgb_to_hsv_np(arr)
            hsv[:, :, 0] = (hsv[:, :, 0] + custom_hue) % 1.0
            rgb = hsv_to_rgb_np(hsv)
            shifted_img = Image.fromarray((rgb * 255.0).astype(np.uint8))
            shift_label = f"+{int(custom_hue * 360)}° Custom Palette Shift"
        else:
            shifted_img = generate_colorway_1(img)
            shift_label = "Standard Shift"
            
        emb_orig = embed_image(img, model, transform, DEVICE)
        emb_shifted = embed_image(shifted_img, model, transform, DEVICE)
        
        similarity = float(np.dot(emb_orig, emb_shifted))
        
        return jsonify({
            'success': True,
            'shift_label': shift_label,
            'similarity': round(similarity, 4),
            'threshold': FROZEN_VAL_EER_THRESHOLD,
            'is_match': similarity >= FROZEN_VAL_EER_THRESHOLD,
            'margin': round(similarity - FROZEN_VAL_EER_THRESHOLD, 4),
            'original_b64': f"data:image/jpeg;base64,{pil_to_base64(img)}",
            'shifted_b64': f"data:image/jpeg;base64,{pil_to_base64(shifted_img)}",
            'verdict': "INVARIANT MATCH" if similarity >= FROZEN_VAL_EER_THRESHOLD else "DIVERGENT"
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MotifWeave | Color-Invariant Textile Design Recognition & Retrieval</title>
  <meta name="description" content="AI platform recognizing saree motifs and weave patterns invariant to colorway palette shifts using ConvNeXt-Tiny, GeM Pooling, and Supervised Contrastive Embeddings." />
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0a0d14;
      --bg-card: rgba(18, 24, 38, 0.75);
      --bg-card-hover: rgba(26, 34, 53, 0.9);
      --bg-glass: rgba(255, 255, 255, 0.03);
      --border-card: rgba(255, 255, 255, 0.08);
      --border-active: rgba(99, 102, 241, 0.5);
      --primary: #6366f1;
      --primary-glow: rgba(99, 102, 241, 0.35);
      --accent: #8b5cf6;
      --amber: #f59e0b;
      --emerald: #10b981;
      --emerald-glow: rgba(16, 185, 129, 0.3);
      --rose: #f43f5e;
      --rose-glow: rgba(244, 63, 94, 0.3);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 20px;
      --font-display: 'Outfit', sans-serif;
      --font-body: 'Plus Jakarta Sans', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg-dark);
      background-image: 
        radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 85% 25%, rgba(139, 92, 246, 0.12) 0%, transparent 45%),
        radial-gradient(circle at 50% 90%, rgba(245, 158, 11, 0.08) 0%, transparent 50%);
      background-attachment: fixed;
      color: var(--text-main);
      font-family: var(--font-body);
      min-height: 100vh;
      line-height: 1.5;
      overflow-x: hidden;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }
    ::-webkit-scrollbar-track {
      background: var(--bg-dark);
    }
    ::-webkit-scrollbar-thumb {
      background: #273142;
      border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: #3e4c63;
    }

    /* Header */
    header {
      border-bottom: 1px solid var(--border-card);
      backdrop-filter: blur(20px);
      background: rgba(10, 13, 20, 0.85);
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .nav-container {
      max-width: 1400px;
      margin: 0 auto;
      padding: 16px 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
      text-decoration: none;
    }

    .logo-badge {
      width: 44px;
      height: 44px;
      border-radius: var(--radius-md);
      background: linear-gradient(135deg, var(--primary), var(--accent));
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 20px var(--primary-glow);
    }

    .logo-badge svg {
      width: 24px;
      height: 24px;
      color: white;
    }

    .brand-text h1 {
      font-family: var(--font-display);
      font-size: 1.35rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(to right, #fff, #cbd5e1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand-text p {
      font-size: 0.75rem;
      color: var(--text-dim);
      font-family: var(--font-mono);
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }

    .header-pills {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.3);
      border-radius: 9999px;
      font-size: 0.8rem;
      font-family: var(--font-mono);
      color: #34d399;
    }

    .status-indicator {
      width: 8px;
      height: 8px;
      background: #10b981;
      border-radius: 50%;
      box-shadow: 0 0 10px #10b981;
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Main Container */
    main {
      max-width: 1400px;
      margin: 0 auto;
      padding: 32px 28px 80px;
    }

    /* Hero / Stat Ribbon */
    .hero-ribbon {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 32px;
    }

    .stat-card {
      background: var(--bg-card);
      border: 1px solid var(--border-card);
      border-radius: var(--radius-md);
      padding: 16px 20px;
      backdrop-filter: blur(12px);
      transition: all 0.25s ease;
      position: relative;
      overflow: hidden;
    }

    .stat-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 2px;
      background: linear-gradient(90deg, transparent, var(--primary), transparent);
      opacity: 0;
      transition: opacity 0.3s ease;
    }

    .stat-card:hover {
      border-color: var(--border-active);
      transform: translateY(-2px);
    }

    .stat-card:hover::before {
      opacity: 1;
    }

    .stat-label {
      font-size: 0.75rem;
      color: var(--text-dim);
      font-family: var(--font-mono);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
    }

    .stat-value {
      font-family: var(--font-display);
      font-size: 1.5rem;
      font-weight: 700;
      color: #fff;
    }

    .stat-sub {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 4px;
    }

    /* Tabs Navigation */
    .tabs-nav {
      display: flex;
      gap: 8px;
      background: rgba(18, 24, 38, 0.6);
      padding: 6px;
      border-radius: var(--radius-md);
      border: 1px solid var(--border-card);
      margin-bottom: 28px;
      overflow-x: auto;
    }

    .tab-btn {
      padding: 10px 22px;
      background: transparent;
      border: none;
      border-radius: var(--radius-sm);
      color: var(--text-muted);
      font-family: var(--font-display);
      font-size: 0.92rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 10px;
      white-space: nowrap;
      transition: all 0.2s ease;
    }

    .tab-btn:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.04);
    }

    .tab-btn.active {
      color: #fff;
      background: linear-gradient(135deg, rgba(99, 102, 241, 0.8), rgba(139, 92, 246, 0.8));
      box-shadow: 0 4px 16px var(--primary-glow);
    }

    /* Tab Content Areas */
    .tab-content {
      display: none;
      animation: fadeIn 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .tab-content.active {
      display: block;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Panel layout */
    .dual-panel {
      display: grid;
      grid-template-columns: 440px 1fr;
      gap: 28px;
    }

    @media (max-width: 1024px) {
      .dual-panel {
        grid-template-columns: 1fr;
      }
    }

    .panel {
      background: var(--bg-card);
      border: 1px solid var(--border-card);
      border-radius: var(--radius-lg);
      padding: 24px;
      backdrop-filter: blur(16px);
    }

    .panel-header {
      margin-bottom: 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .panel-title {
      font-family: var(--font-display);
      font-size: 1.15rem;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .panel-subtitle {
      font-size: 0.82rem;
      color: var(--text-dim);
      margin-top: 4px;
    }

    /* Upload Box */
    .upload-box {
      border: 2px dashed rgba(255, 255, 255, 0.12);
      border-radius: var(--radius-md);
      padding: 28px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.25s ease;
      background: rgba(255, 255, 255, 0.015);
      position: relative;
    }

    .upload-box:hover, .upload-box.dragover {
      border-color: var(--primary);
      background: rgba(99, 102, 241, 0.06);
    }

    .upload-box input[type="file"] {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      opacity: 0;
      cursor: pointer;
    }

    .upload-icon {
      width: 48px;
      height: 48px;
      margin: 0 auto 12px;
      color: var(--primary);
    }

    .preview-box {
      margin-top: 18px;
      border-radius: var(--radius-md);
      overflow: hidden;
      border: 1px solid var(--border-card);
      position: relative;
      max-height: 280px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #000;
    }

    .preview-box img {
      width: 100%;
      height: 280px;
      object-fit: cover;
      display: block;
    }

    .preview-badge {
      position: absolute;
      bottom: 10px;
      left: 10px;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(8px);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-family: var(--font-mono);
      color: #cbd5e1;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Sample Selector Grid */
    .sample-section {
      margin-top: 24px;
    }

    .sample-title {
      font-size: 0.8rem;
      font-family: var(--font-mono);
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
    }

    .sample-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
    }

    .sample-card {
      border: 1px solid var(--border-card);
      border-radius: var(--radius-sm);
      overflow: hidden;
      cursor: pointer;
      transition: all 0.2s ease;
      background: #0f1522;
      position: relative;
    }

    .sample-card:hover {
      border-color: var(--primary);
      transform: scale(1.03);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    }

    .sample-card.active {
      border-color: var(--primary);
      box-shadow: 0 0 12px var(--primary-glow);
    }

    .sample-card img {
      width: 100%;
      height: 68px;
      object-fit: cover;
      display: block;
    }

    .sample-tag {
      font-size: 0.65rem;
      padding: 3px 6px;
      text-align: center;
      background: rgba(18, 24, 38, 0.9);
      color: var(--text-muted);
      font-family: var(--font-mono);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    /* Buttons & Controls */
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 12px 24px;
      border-radius: var(--radius-md);
      font-family: var(--font-display);
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      border: none;
      width: 100%;
      text-decoration: none;
    }

    .btn-primary {
      background: linear-gradient(135deg, var(--primary), var(--accent));
      color: #fff;
      box-shadow: 0 4px 16px var(--primary-glow);
    }

    .btn-primary:hover {
      box-shadow: 0 6px 24px rgba(99, 102, 241, 0.5);
      transform: translateY(-1px);
    }

    .btn-secondary {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border-card);
      color: var(--text-main);
    }

    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.2);
    }

    /* Results Grid */
    .results-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 18px;
    }

    .result-card {
      background: var(--bg-card);
      border: 1px solid var(--border-card);
      border-radius: var(--radius-md);
      overflow: hidden;
      transition: all 0.25s ease;
      position: relative;
    }

    .result-card:hover {
      border-color: var(--border-active);
      transform: translateY(-3px);
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.4);
    }

    .result-card.is-top {
      border-color: rgba(16, 185, 129, 0.6);
      box-shadow: 0 0 20px var(--emerald-glow);
    }

    .result-rank {
      position: absolute;
      top: 10px;
      left: 10px;
      width: 28px;
      height: 28px;
      border-radius: 50%;
      background: rgba(0, 0, 0, 0.8);
      backdrop-filter: blur(8px);
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: var(--font-display);
      font-size: 0.85rem;
      font-weight: 700;
      color: #fff;
      border: 1px solid rgba(255, 255, 255, 0.2);
      z-index: 2;
    }

    .result-card.is-top .result-rank {
      background: var(--emerald);
      border-color: #34d399;
    }

    .result-img {
      width: 100%;
      height: 200px;
      object-fit: cover;
      display: block;
      background: #000;
      transition: transform 0.4s ease;
    }

    .result-card:hover .result-img {
      transform: scale(1.04);
    }

    .result-info {
      padding: 16px;
    }

    .result-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }

    .result-source {
      font-family: var(--font-mono);
      font-size: 0.8rem;
      color: #cbd5e1;
    }

    .similarity-pill {
      font-family: var(--font-mono);
      font-size: 0.85rem;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 9999px;
      background: rgba(99, 102, 241, 0.15);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .result-card.is-top .similarity-pill {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border-color: rgba(16, 185, 129, 0.3);
    }

    .score-bar-bg {
      height: 6px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 9999px;
      overflow: hidden;
      margin-top: 8px;
    }

    .score-bar-fill {
      height: 100%;
      border-radius: 9999px;
      background: linear-gradient(90deg, var(--primary), var(--accent));
      transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .result-card.is-top .score-bar-fill {
      background: linear-gradient(90deg, #10b981, #34d399);
    }

    /* Verification Section Styles */
    .verification-container {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-bottom: 28px;
    }

    @media (max-width: 768px) {
      .verification-container {
        grid-template-columns: 1fr;
      }
    }

    .compare-card {
      background: var(--bg-card);
      border: 1px solid var(--border-card);
      border-radius: var(--radius-lg);
      padding: 20px;
      text-align: center;
    }

    .gauge-wrapper {
      margin: 32px auto;
      text-align: center;
      position: relative;
      max-width: 320px;
    }

    .gauge-svg {
      width: 240px;
      height: 140px;
      overflow: visible;
    }

    .gauge-value {
      position: absolute;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      font-family: var(--font-display);
      font-size: 2.2rem;
      font-weight: 800;
      color: #fff;
    }

    .verdict-banner {
      padding: 20px 28px;
      border-radius: var(--radius-md);
      text-align: center;
      font-family: var(--font-display);
      font-size: 1.4rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      margin-top: 20px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 14px;
      transition: all 0.3s ease;
    }

    .verdict-banner.match {
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: #34d399;
      box-shadow: 0 0 30px var(--emerald-glow);
    }

    .verdict-banner.no-match {
      background: rgba(244, 63, 94, 0.12);
      border: 1px solid rgba(244, 63, 94, 0.4);
      color: #fb7185;
      box-shadow: 0 0 30px var(--rose-glow);
    }

    /* Simulator Styles */
    .simulator-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 28px;
      margin-top: 24px;
    }

    @media (max-width: 768px) {
      .simulator-grid {
        grid-template-columns: 1fr;
      }
    }

    .sim-img-card {
      background: var(--bg-card);
      border: 1px solid var(--border-card);
      border-radius: var(--radius-md);
      overflow: hidden;
      padding: 16px;
      text-align: center;
    }

    .sim-img-card img {
      width: 100%;
      height: 320px;
      object-fit: cover;
      border-radius: var(--radius-sm);
      display: block;
      margin-bottom: 12px;
      background: #000;
    }

    /* Empty state & loaders */
    .empty-state {
      padding: 60px 20px;
      text-align: center;
      color: var(--text-dim);
    }

    .spinner {
      display: inline-block;
      width: 20px;
      height: 20px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 50%;
      border-top-color: #fff;
      animation: spin 0.8s ease-in-out infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    /* Modal */
    .modal-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(0, 0, 0, 0.85);
      backdrop-filter: blur(12px);
      z-index: 1000;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }

    .modal-overlay.active {
      display: flex;
    }

    .modal-box {
      background: #0f1523;
      border: 1px solid var(--border-card);
      border-radius: var(--radius-lg);
      max-width: 900px;
      width: 100%;
      padding: 28px;
      position: relative;
    }

    .modal-close {
      position: absolute;
      top: 20px;
      right: 20px;
      background: transparent;
      border: none;
      color: var(--text-dim);
      font-size: 1.5rem;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <!-- Top Navigation Header -->
  <header>
    <div class="nav-container">
      <a href="/" class="brand">
        <div class="logo-badge">
          <svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 21a9.004 9.004 0 008.716-6.747M12 21a9.004 9.004 0 01-8.716-6.747M12 21c2.485 0 4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997 8.997 0 017.843 4.582M12 3a8.997 8.997 0 00-7.843 4.582m15.686 0A11.953 11.953 0 0112 10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0121 12c0 .778-.099 1.533-.284 2.253m0 0A17.919 17.919 0 0112 16.5c-3.162 0-6.133-.815-8.716-2.247m0 0A9.015 9.015 0 013 12c0-.778.099-1.533.284-2.253" />
          </svg>
        </div>
        <div class="brand-text">
          <h1>MotifWeave AI</h1>
          <p>Color-Invariant Saree Design Recognition</p>
        </div>
      </a>

      <div class="header-pills">
        <div class="status-pill">
          <div class="status-indicator"></div>
          <span>ConvNeXt-Tiny (GeM p=2.98)</span>
        </div>
        <div class="status-pill" style="border-color: rgba(99, 102, 241, 0.3); color: #a5b4fc; background: rgba(99, 102, 241, 0.1);">
          <span>256-D Hypersphere</span>
        </div>
      </div>
    </div>
  </header>

  <main>
    <!-- Hero / Stat Ribbon -->
    <div class="hero-ribbon">
      <div class="stat-card">
        <div class="stat-label">Recall @ 1 (Held-out)</div>
        <div class="stat-value" style="color: #34d399;">97.50%</div>
        <div class="stat-sub">117 / 120 Queries retrieved rank-1</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Mean Average Precision</div>
        <div class="stat-value" style="color: #a5b4fc;">98.75%</div>
        <div class="stat-sub">mAP across triadic color shifts</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">ROC - AUC</div>
        <div class="stat-value" style="color: #fcd34d;">0.9967</div>
        <div class="stat-sub">Test EER proxy: 1.31%</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Separation Margin</div>
        <div class="stat-value" style="color: #38bdf8;">+0.5966</div>
        <div class="stat-sub">Pos: 0.8894 vs Neg: 0.2928</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Operating Threshold</div>
        <div class="stat-value" style="color: #f472b6;">0.7588</div>
        <div class="stat-sub">Frozen Val EER calibration</div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="tabs-nav">
      <button class="tab-btn active" onclick="switchTab('retrieval')">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
        Visual Motif Retrieval
      </button>
      <button class="tab-btn" onclick="switchTab('verification')">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
        Pairwise Verification
      </button>
      <button class="tab-btn" onclick="switchTab('simulator')">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01"/></svg>
        Colorway Shift Simulator
      </button>
      <button class="tab-btn" onclick="switchTab('gallery')">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"/></svg>
        Reference Gallery (40 Sarees)
      </button>
    </div>

    <!-- TAB 1: VISUAL RETRIEVAL -->
    <div id="tab-retrieval" class="tab-content active">
      <div class="dual-panel">
        <!-- Left: Query Input Panel -->
        <div class="panel">
          <div class="panel-header">
            <div>
              <div class="panel-title">Query Image</div>
              <div class="panel-subtitle">Upload saree photo or select from test colorways</div>
            </div>
          </div>

          <div class="upload-box" id="queryDropzone">
            <input type="file" id="queryFileInput" accept="image/*" onchange="handleQueryFileUpload(event)" />
            <svg class="upload-icon" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/></svg>
            <p style="font-weight: 600; font-size: 0.95rem;">Drag & drop image here</p>
            <p style="color: var(--text-dim); font-size: 0.8rem; margin-top: 4px;">JPEG, PNG or WEBP (automatically resized to 224x224)</p>
          </div>

          <div class="preview-box" id="queryPreviewBox" style="display: none;">
            <img id="queryPreviewImg" src="" alt="Query Preview" />
            <div class="preview-badge" id="queryPreviewBadge">Selected Image</div>
          </div>

          <!-- Quick Test Samples -->
          <div class="sample-section">
            <div class="sample-title">
              <span>Quick Test Samples</span>
              <span style="color: var(--primary);">Click to test color invariance</span>
            </div>
            <div class="sample-grid" id="querySampleGrid">
              <!-- Dynamically populated -->
            </div>
          </div>

          <div style="margin-top: 24px;">
            <div style="display: flex; justify-content: space-between; font-size: 0.82rem; color: var(--text-muted); margin-bottom: 8px;">
              <span>Top-K Candidates:</span>
              <span id="topKValue" style="font-family: var(--font-mono); color: #fff;">6</span>
            </div>
            <input type="range" id="topKSlider" min="1" max="15" value="6" style="width: 100%; accent-color: var(--primary); margin-bottom: 20px;" oninput="document.getElementById('topKValue').innerText = this.value" />

            <button class="btn btn-primary" id="btnRunQuery" onclick="executeRetrieval()">
              <span id="btnRunQueryText">Search Gallery for Matches</span>
            </button>
          </div>
        </div>

        <!-- Right: Results Panel -->
        <div class="panel">
          <div class="panel-header">
            <div>
              <div class="panel-title">Top-K Nearest Designs in Gallery</div>
              <div class="panel-subtitle">Calculated via exact dot-product cosine similarity on 256-D hypersphere</div>
            </div>
            <div id="resultsCountBadge" style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-dim);"></div>
          </div>

          <div id="resultsLoading" style="display: none;" class="empty-state">
            <div class="spinner" style="width: 32px; height: 32px; border-width: 3px; border-top-color: var(--primary); margin-bottom: 16px;"></div>
            <p style="font-family: var(--font-display); font-size: 1.1rem; color: #fff;">Computing 256-D Embedding Vector...</p>
            <p style="font-size: 0.85rem; color: var(--text-dim); margin-top: 4px;">Evaluating ConvNeXt-Tiny + GeM pooling across reference index</p>
          </div>

          <div id="resultsEmpty" class="empty-state">
            <svg style="width: 56px; height: 56px; color: var(--text-dim); margin-bottom: 12px; opacity: 0.6;" fill="none" stroke="currentColor" stroke-width="1.2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z"/></svg>
            <p style="font-size: 1.05rem; font-weight: 600; color: #cbd5e1;">Select a sample image or upload a photo</p>
            <p style="font-size: 0.85rem; color: var(--text-dim); margin-top: 4px;">The model will retrieve identical textile weaves across completely different colorways.</p>
          </div>

          <div id="resultsGrid" class="results-grid" style="display: none;">
            <!-- Retrieved items dynamically inserted -->
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: PAIRWISE VERIFICATION -->
    <div id="tab-verification" class="tab-content">
      <div class="panel" style="margin-bottom: 24px;">
        <div class="panel-header">
          <div>
            <div class="panel-title">Pairwise Design Verification Benchmark</div>
            <div class="panel-subtitle">Evaluate whether two sarees share the exact same structural motif using pre-calibrated thresholds</div>
          </div>
          <div style="display: flex; gap: 8px;">
            <button class="btn btn-secondary" style="padding: 8px 16px; font-size: 0.82rem;" onclick="loadPresetPair('matching')">Load Same-Design Pair (+120°)</button>
            <button class="btn btn-secondary" style="padding: 8px 16px; font-size: 0.82rem;" onclick="loadPresetPair('distinct')">Load Distinct-Design Pair</button>
          </div>
        </div>

        <div class="verification-container">
          <!-- Image A -->
          <div class="compare-card">
            <h3 style="font-family: var(--font-display); font-size: 1rem; color: #fff; margin-bottom: 12px;">Design Image A</h3>
            <div class="preview-box" style="height: 240px;">
              <img id="verifyImgA" src="/media/data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/h_img_149526.jpg" alt="Image A" />
              <div class="preview-badge" id="badgeA">deeplure_002 (Original)</div>
            </div>
            <div style="margin-top: 14px;">
              <input type="file" id="fileA" accept="image/*" style="display: none;" onchange="handleVerifyUpload(event, 'A')" />
              <button class="btn btn-secondary" style="font-size: 0.82rem; padding: 8px;" onclick="document.getElementById('fileA').click()">Upload Custom Image A</button>
            </div>
          </div>

          <!-- Image B -->
          <div class="compare-card">
            <h3 style="font-family: var(--font-display); font-size: 1rem; color: #fff; margin-bottom: 12px;">Design Image B</h3>
            <div class="preview-box" style="height: 240px;">
              <img id="verifyImgB" src="/media/data/processed/variants/deeplure_002_variant1.jpg" alt="Image B" />
              <div class="preview-badge" id="badgeB">deeplure_002 (+120° Triadic)</div>
            </div>
            <div style="margin-top: 14px;">
              <input type="file" id="fileB" accept="image/*" style="display: none;" onchange="handleVerifyUpload(event, 'B')" />
              <button class="btn btn-secondary" style="font-size: 0.82rem; padding: 8px;" onclick="document.getElementById('fileB').click()">Upload Custom Image B</button>
            </div>
          </div>
        </div>

        <!-- Controls & Gauge -->
        <div style="display: flex; justify-content: center; gap: 20px; align-items: center; margin-bottom: 20px;">
          <label style="font-size: 0.85rem; color: var(--text-muted); display: flex; align-items: center; gap: 8px; cursor: pointer;">
            <input type="radio" name="threshMode" value="eer" checked onchange="runPairwiseVerification()" />
            <span>Validation EER Threshold (&theta; = 0.7588)</span>
          </label>
          <label style="font-size: 0.85rem; color: var(--text-muted); display: flex; align-items: center; gap: 8px; cursor: pointer;">
            <input type="radio" name="threshMode" value="max_f1" onchange="runPairwiseVerification()" />
            <span>Validation Max-F1 Threshold (&theta; = 0.8359)</span>
          </label>
        </div>

        <div style="max-width: 380px; margin: 0 auto;">
          <button class="btn btn-primary" onclick="runPairwiseVerification()">
            <span id="btnVerifyText">Run Verification Analysis</span>
          </button>
        </div>

        <!-- Verification Results Presentation -->
        <div id="verifyResultsBox" style="margin-top: 32px; display: none;">
          <div style="display: grid; grid-template-columns: 280px 1fr; gap: 32px; align-items: center;">
            <div class="gauge-wrapper">
              <svg class="gauge-svg" viewBox="0 0 200 110">
                <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="16" stroke-linecap="round" />
                <path id="gaugeArc" d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="url(#gaugeGrad)" stroke-width="16" stroke-linecap="round" stroke-dasharray="251.2" stroke-dashoffset="251.2" style="transition: stroke-dashoffset 0.8s ease;" />
                <line id="threshLine" x1="100" y1="20" x2="100" y2="40" stroke="#fcd34d" stroke-width="3" stroke-dasharray="2,2" />
                <defs>
                  <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#f43f5e" />
                    <stop offset="60%" stop-color="#f59e0b" />
                    <stop offset="100%" stop-color="#10b981" />
                  </linearGradient>
                </defs>
              </svg>
              <div class="gauge-value" id="gaugeValText">0.0000</div>
              <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-dim); margin-top: -12px;">Cosine Similarity</div>
            </div>

            <div>
              <div id="verdictBanner" class="verdict-banner match">
                <svg id="verdictIcon" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>
                <span id="verdictText">SAME DESIGN (MATCH)</span>
              </div>

              <div style="margin-top: 16px; background: rgba(0,0,0,0.3); border-radius: var(--radius-sm); padding: 16px; border: 1px solid var(--border-card);">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-family: var(--font-mono); font-size: 0.82rem;">
                  <div><span style="color: var(--text-dim);">Operating Threshold:</span> <strong id="metricThresh">0.7588</strong></div>
                  <div><span style="color: var(--text-dim);">Distance Margin:</span> <strong id="metricMargin">+0.1060</strong></div>
                  <div><span style="color: var(--text-dim);">Val EER Decision:</span> <strong id="metricEERDecision">SAME</strong></div>
                  <div><span style="color: var(--text-dim);">Val Max-F1 Decision:</span> <strong id="metricF1Decision">SAME</strong></div>
                </div>
                <div id="metricExplanation" style="margin-top: 12px; font-size: 0.85rem; color: var(--text-muted); line-height: 1.4;"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: COLORWAY SHIFT SIMULATOR -->
    <div id="tab-simulator" class="tab-content">
      <div class="panel">
        <div class="panel-header">
          <div>
            <div class="panel-title">Colorway Invariance Laboratory</div>
            <div class="panel-subtitle">Demonstrate live color transformations and prove embedding stability on identical motifs</div>
          </div>
        </div>

        <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px;">
          <button class="btn btn-secondary" style="width: auto; padding: 8px 18px;" onclick="runSimulation('variant1')">+120° Triadic Hue Shift</button>
          <button class="btn btn-secondary" style="width: auto; padding: 8px 18px;" onclick="runSimulation('variant2')">+240° Triadic Hue Shift</button>
          <button class="btn btn-secondary" style="width: auto; padding: 8px 18px;" onclick="runSimulation('variant3')">+50° Antique Dye Shift</button>
          <button class="btn btn-secondary" style="width: auto; padding: 8px 18px;" onclick="runSimulation('grayscale')">Achromatic Grayscale</button>
        </div>

        <div class="simulator-grid">
          <div class="sim-img-card">
            <h4 style="font-family: var(--font-display); color: #fff; margin-bottom: 8px;">Original Base Saree</h4>
            <img id="simOrigImg" src="/media/data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/h_img_149526.jpg" alt="Original Saree" />
            <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-dim);">Source: deeplure_002 (Golden Brocade)</div>
          </div>

          <div class="sim-img-card">
            <h4 style="font-family: var(--font-display); color: #fff; margin-bottom: 8px;">Transformed Colorway</h4>
            <img id="simShiftedImg" src="/media/data/processed/variants/deeplure_002_variant1.jpg" alt="Shifted Saree" />
            <div style="font-family: var(--font-mono); font-size: 0.8rem; color: #a5b4fc;" id="simShiftLabel">+120° Triadic Hue Shift (Synthetic)</div>
          </div>
        </div>

        <!-- Simulation metric output -->
        <div id="simMetricBox" style="margin-top: 24px; padding: 20px; background: rgba(0,0,0,0.35); border-radius: var(--radius-md); border: 1px solid var(--border-card); display: flex; justify-content: space-around; align-items: center;">
          <div>
            <div style="font-size: 0.75rem; font-family: var(--font-mono); color: var(--text-dim);">EMBEDDING SIMILARITY</div>
            <div style="font-family: var(--font-display); font-size: 1.8rem; font-weight: 800; color: #34d399;" id="simSimScore">0.8648</div>
          </div>
          <div>
            <div style="font-size: 0.75rem; font-family: var(--font-mono); color: var(--text-dim);">EER DECISION THRESHOLD</div>
            <div style="font-family: var(--font-display); font-size: 1.8rem; font-weight: 800; color: #fff;">0.7588</div>
          </div>
          <div>
            <div style="font-size: 0.75rem; font-family: var(--font-mono); color: var(--text-dim);">MOTIF PRESERVATION</div>
            <div style="font-family: var(--font-display); font-size: 1.8rem; font-weight: 800; color: #34d399;" id="simVerdict">INVARIANT MATCH</div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: REFERENCE GALLERY -->
    <div id="tab-gallery" class="tab-content">
      <div class="panel">
        <div class="panel-header">
          <div>
            <div class="panel-title">Authoritative 40-Source Reference Gallery</div>
            <div class="panel-subtitle">Pre-computed L2-normalized 256-D embedding index without duplicate image bytes</div>
          </div>
          <div style="font-family: var(--font-mono); font-size: 0.85rem; color: #34d399;">
            40 Unique Design Sources
          </div>
        </div>

        <div class="results-grid" id="fullGalleryGrid">
          <!-- Populated dynamically -->
        </div>
      </div>
    </div>
  </main>

  <script>
    let currentQueryPath = 'data/processed/variants/deeplure_002_variant1.jpg';
    let currentVerifyA = 'data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/h_img_149526.jpg';
    let currentVerifyB = 'data/processed/variants/deeplure_002_variant1.jpg';
    let uploadedQueryFile = null;
    let uploadedFileA = null;
    let uploadedFileB = null;

    // Tab Switching
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      const targetBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
      if (targetBtn) targetBtn.classList.add('active');
      
      const targetContent = document.getElementById('tab-' + tabId);
      if (targetContent) targetContent.classList.add('active');

      if (tabId === 'gallery') {
        loadFullGallery();
      }
    }

    // Load Samples on Startup
    async function loadSampleImages() {
      try {
        const res = await fetch('/api/samples');
        const samples = await res.json();
        const grid = document.getElementById('querySampleGrid');
        grid.innerHTML = '';
        
        let first = true;
        samples.forEach(s => {
          // Add variant 1 (color-shifted)
          const div1 = document.createElement('div');
          div1.className = 'sample-card' + (first ? ' active' : '');
          div1.onclick = () => selectQuerySample(s.variant1.path, s.variant1.url, s.variant1.title, div1);
          div1.innerHTML = `<img src="${s.variant1.url}" alt="${s.variant1.title}" /><div class="sample-tag">${s.source_id} +120°</div>`;
          grid.appendChild(div1);

          // Add variant 2
          const div2 = document.createElement('div');
          div2.className = 'sample-card';
          div2.onclick = () => selectQuerySample(s.variant2.path, s.variant2.url, s.variant2.title, div2);
          div2.innerHTML = `<img src="${s.variant2.url}" alt="${s.variant2.title}" /><div class="sample-tag">${s.source_id} +240°</div>`;
          grid.appendChild(div2);

          if (first) {
            selectQuerySample(s.variant1.path, s.variant1.url, s.variant1.title, div1);
            first = false;
          }
        });
      } catch (err) {
        console.error('Failed to load samples:', err);
      }
    }

    function selectQuerySample(path, url, title, cardElem) {
      currentQueryPath = path;
      uploadedQueryFile = null;
      document.querySelectorAll('#querySampleGrid .sample-card').forEach(c => c.classList.remove('active'));
      if (cardElem) cardElem.classList.add('active');

      const previewBox = document.getElementById('queryPreviewBox');
      const previewImg = document.getElementById('queryPreviewImg');
      const badge = document.getElementById('queryPreviewBadge');

      previewImg.src = url;
      badge.innerText = title;
      previewBox.style.display = 'flex';
      
      // Auto-trigger retrieval for fluid experience
      executeRetrieval();
    }

    function handleQueryFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;
      uploadedQueryFile = file;
      currentQueryPath = null;
      document.querySelectorAll('#querySampleGrid .sample-card').forEach(c => c.classList.remove('active'));

      const reader = new FileReader();
      reader.onload = (e) => {
        const previewBox = document.getElementById('queryPreviewBox');
        const previewImg = document.getElementById('queryPreviewImg');
        const badge = document.getElementById('queryPreviewBadge');
        previewImg.src = e.target.result;
        badge.innerText = 'Uploaded: ' + file.name;
        previewBox.style.display = 'flex';
        executeRetrieval();
      };
      reader.readAsDataURL(file);
    }

    // Execute Retrieval
    async function executeRetrieval() {
      const loading = document.getElementById('resultsLoading');
      const empty = document.getElementById('resultsEmpty');
      const grid = document.getElementById('resultsGrid');
      const btnText = document.getElementById('btnRunQueryText');
      const topK = document.getElementById('topKSlider').value;

      loading.style.display = 'block';
      empty.style.display = 'none';
      grid.style.display = 'none';
      btnText.innerHTML = '<span class="spinner"></span> Searching...';

      const formData = new FormData();
      formData.append('top_k', topK);

      if (uploadedQueryFile) {
        formData.append('query_image', uploadedQueryFile);
      } else if (currentQueryPath) {
        formData.append('query_path', currentQueryPath);
      } else {
        alert('Please choose or upload a query image.');
        loading.style.display = 'none';
        btnText.innerText = 'Search Gallery for Matches';
        return;
      }

      try {
        const res = await fetch('/api/query', { method: 'POST', body: formData });
        const data = await res.json();
        loading.style.display = 'none';
        btnText.innerText = 'Search Gallery for Matches';

        if (!data.success) {
          alert('Retrieval error: ' + data.error);
          return;
        }

        renderResults(data.results);
      } catch (err) {
        loading.style.display = 'none';
        btnText.innerText = 'Search Gallery for Matches';
        alert('Request failed: ' + err);
      }
    }

    function renderResults(results) {
      const grid = document.getElementById('resultsGrid');
      const countBadge = document.getElementById('resultsCountBadge');
      grid.innerHTML = '';
      countBadge.innerText = results.length + ' Candidates Evaluated';

      results.forEach((item, index) => {
        const card = document.createElement('div');
        const isTop = index === 0;
        card.className = 'result-card' + (isTop ? ' is-top' : '');

        const percent = Math.max(0, Math.min(100, Math.round(item.similarity * 100)));

        card.innerHTML = `
          <div class="result-rank">${item.rank}</div>
          <img class="result-img" src="${item.url}" alt="${item.filename}" />
          <div class="result-info">
            <div class="result-meta">
              <span class="result-source">${item.source_id}</span>
              <span class="similarity-pill">${item.similarity.toFixed(4)}</span>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-dim); margin-bottom: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
              ${item.filename}
            </div>
            <div class="score-bar-bg">
              <div class="score-bar-fill" style="width: ${percent}%;"></div>
            </div>
            <div style="margin-top: 12px; display: flex; gap: 8px;">
              <button class="btn btn-secondary" style="padding: 6px 12px; font-size: 0.75rem;" onclick="compareWithQuery('${item.path}', '${item.url}', '${item.source_id}')">
                Compare in Pairwise
              </button>
            </div>
          </div>
        `;
        grid.appendChild(card);
      });

      grid.style.display = 'grid';
    }

    // Pairwise Verification
    async function runPairwiseVerification() {
      const btnText = document.getElementById('btnVerifyText');
      btnText.innerHTML = '<span class="spinner"></span> Verifying...';

      const mode = document.querySelector('input[name="threshMode"]:checked').value;
      const formData = new FormData();
      formData.append('threshold_mode', mode);

      if (uploadedFileA) {
        formData.append('image_a', uploadedFileA);
      } else {
        formData.append('image_a_path', currentVerifyA);
      }

      if (uploadedFileB) {
        formData.append('image_b', uploadedFileB);
      } else {
        formData.append('image_b_path', currentVerifyB);
      }

      try {
        const res = await fetch('/api/verify', { method: 'POST', body: formData });
        const data = await res.json();
        btnText.innerText = 'Run Verification Analysis';

        if (!data.success) {
          alert('Verification error: ' + data.error);
          return;
        }

        renderVerificationResult(data);
      } catch (err) {
        btnText.innerText = 'Run Verification Analysis';
        alert('Verification request failed: ' + err);
      }
    }

    function renderVerificationResult(data) {
      const box = document.getElementById('verifyResultsBox');
      box.style.display = 'block';

      // Gauge animation (similarity between 0 and 1 maps to arc)
      const sim = Math.max(0, Math.min(1, data.similarity));
      const totalLen = 251.2;
      const offset = totalLen - (sim * totalLen);
      document.getElementById('gaugeArc').style.strokeDashoffset = offset;
      document.getElementById('gaugeValText').innerText = data.similarity.toFixed(4);

      // Threshold line rotation
      const threshAngle = (data.threshold * 180) - 90;
      // Banner
      const banner = document.getElementById('verdictBanner');
      const vText = document.getElementById('verdictText');
      const vIcon = document.getElementById('verdictIcon');

      if (data.is_same) {
        banner.className = 'verdict-banner match';
        vText.innerText = 'SAME DESIGN (MATCH)';
        vIcon.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/>';
      } else {
        banner.className = 'verdict-banner no-match';
        vText.innerText = 'DIFFERENT DESIGN (DO NOT MATCH)';
        vIcon.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>';
      }

      document.getElementById('metricThresh').innerText = data.threshold.toFixed(4);
      document.getElementById('metricMargin').innerText = (data.margin >= 0 ? '+' : '') + data.margin.toFixed(4);
      document.getElementById('metricEERDecision').innerText = data.eer_decision;
      document.getElementById('metricF1Decision').innerText = data.f1_decision;
      document.getElementById('metricExplanation').innerText = data.summary + 
        (data.is_same ? ' The geometric motif, weave layout, and brocade border structural features match identically despite chromatic differences.' 
                      : ' The visual textural features, motifs, or field brocades differ significantly beyond the calibrated false-match boundary.');
    }

    function loadPresetPair(type) {
      if (type === 'matching') {
        currentVerifyA = 'data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/h_img_149526.jpg';
        currentVerifyB = 'data/processed/variants/deeplure_002_variant1.jpg';
        document.getElementById('verifyImgA').src = '/media/' + currentVerifyA;
        document.getElementById('badgeA').innerText = 'deeplure_002 (Original Ruby/Gold)';
        document.getElementById('verifyImgB').src = '/media/' + currentVerifyB;
        document.getElementById('badgeB').innerText = 'deeplure_002 (+120° Triadic Emerald)';
      } else {
        currentVerifyA = 'data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/h_img_149526.jpg';
        currentVerifyB = 'data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/img_635895.jpg';
        document.getElementById('verifyImgA').src = '/media/' + currentVerifyA;
        document.getElementById('badgeA').innerText = 'deeplure_002 (Floral Brocade)';
        document.getElementById('verifyImgB').src = '/media/' + currentVerifyB;
        document.getElementById('badgeB').innerText = 'deeplure_104 (Diamond Geometric)';
      }
      uploadedFileA = null;
      uploadedFileB = null;
      runPairwiseVerification();
    }

    function compareWithQuery(galleryPath, galleryUrl, sourceId) {
      switchTab('verification');
      currentVerifyA = currentQueryPath;
      currentVerifyB = galleryPath;
      document.getElementById('verifyImgA').src = document.getElementById('queryPreviewImg').src;
      document.getElementById('badgeA').innerText = 'Query Image';
      document.getElementById('verifyImgB').src = galleryUrl;
      document.getElementById('badgeB').innerText = sourceId + ' (Gallery Match)';
      runPairwiseVerification();
    }

    function handleVerifyUpload(event, slot) {
      const file = event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        if (slot === 'A') {
          uploadedFileA = file;
          document.getElementById('verifyImgA').src = e.target.result;
          document.getElementById('badgeA').innerText = 'Custom Image A';
        } else {
          uploadedFileB = file;
          document.getElementById('verifyImgB').src = e.target.result;
          document.getElementById('badgeB').innerText = 'Custom Image B';
        }
        runPairwiseVerification();
      };
      reader.readAsDataURL(file);
    }

    // Colorway Simulator
    async function runSimulation(shiftType) {
      const formData = new FormData();
      formData.append('shift_type', shiftType);
      formData.append('image_path', 'data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/h_img_149526.jpg');

      try {
        const res = await fetch('/api/simulate_shift', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.success) {
          document.getElementById('simShiftedImg').src = data.shifted_b64;
          document.getElementById('simShiftLabel').innerText = data.shift_label;
          document.getElementById('simSimScore').innerText = data.similarity.toFixed(4);
          document.getElementById('simVerdict').innerText = data.verdict;
        }
      } catch (err) {
        console.error('Simulation error:', err);
      }
    }

    // Full Gallery Browser
    async function loadFullGallery() {
      const grid = document.getElementById('fullGalleryGrid');
      if (grid.children.length > 0) return; // already loaded

      try {
        const res = await fetch('/api/gallery');
        const items = await res.json();
        items.forEach(item => {
          const card = document.createElement('div');
          card.className = 'result-card';
          card.innerHTML = `
            <img class="result-img" src="${item.url}" alt="${item.filename}" loading="lazy" />
            <div class="result-info">
              <div class="result-meta">
                <span class="result-source">${item.source_id}</span>
                <span style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-dim);">#${item.index + 1}</span>
              </div>
              <div style="font-size: 0.75rem; color: var(--text-dim); margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                ${item.filename}
              </div>
              <button class="btn btn-secondary" style="padding: 6px 12px; font-size: 0.75rem;" onclick="selectQuerySample('${item.path}', '${item.url}', '${item.source_id}', null); switchTab('retrieval');">
                Search With This
              </button>
            </div>
          `;
          grid.appendChild(card);
        });
      } catch (err) {
        console.error('Failed to load full gallery:', err);
      }
    }

    // Initialization
    window.addEventListener('DOMContentLoaded', () => {
      loadSampleImages();
      runPairwiseVerification();
    });
  </script>
</body>
</html>
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"=================================================================")
    print(f"MOTIFWEAVE INTERACTIVE WEB APPLICATION RUNNING")
    print(f"Local Server URL: http://localhost:{port}")
    print(f"=================================================================")
    app.run(host='0.0.0.0', port=port, debug=False)
