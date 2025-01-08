import os
import cv2
import glob
import numpy as np
import pandas as pd
from scipy import ndimage
import math
from tqdm import tqdm
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import io
import re
import pytesseract
from pytesseract import Output
import easyocr

########################################
# 텍스트 품질 평가 함수
########################################

def evaluate_text_quality(text, valid_words=None):
    """
    (필요 시) 텍스트 품질을 평가하는 함수
    """
    if not text.strip():
        return 0
    
    korean_chars = sum(1 for c in text if '가' <= c <= '힣')
    numbers = sum(1 for c in text if c.isdigit())
    alpha = sum(1 for c in text if c.isalpha() and not ('가' <= c <= '힣'))
    special_chars = len(text) - korean_chars - numbers - alpha
    
    score = (korean_chars * 5) + (numbers * 4) + (alpha * 1.5) + (special_chars * 0.1)
    
    if len(text) > 0:
        meaningful_ratio = (korean_chars + numbers) / len(text)
        score *= (1 + meaningful_ratio)

    # 사전 단어 매칭 점수
    if valid_words is not None and len(valid_words) > 0:
        filtered_text = re.sub("[^가-힣0-9 ]", " ", text)
        words = filtered_text.split()
        dictionary_score = 0
        for w in words:
            if w in valid_words:
                dictionary_score += 10
        score += dictionary_score
    
    return score

########################################
# PaddleOCR 결과로부터 텍스트/신뢰도 추출
# + "수직 텍스트(90, 270)" 제외
########################################

def extract_text_from_tesseract_result(
        result, 
        min_confidence=40, # Tesseract는 0-100 스케일을 사용
        allowed_chars="가-힣0-9",
        debug=False,
        only_horizontal=True
    ):
    """
    Tesseract OCR 결과에서 텍스트와 신뢰도를 추출
    """
    if not result or 'text' not in result or 'conf' not in result:
        return [], []
    
    text_boxes = []
    confidences = []
    
    for i, (text, conf, left, top, width, height) in enumerate(zip(
        result['text'], result['conf'], 
        result['left'], result['top'],
        result['width'], result['height']
    )):
        # 빈 텍스트나 신뢰도가 없는 경우 건너뛰기
        if not text.strip() or conf < min_confidence:
            continue
            
        # 허용된 문자만 필터링
        filtered_text = re.sub(f"[^{allowed_chars}]", " ", text)
        filtered_text = re.sub(" +", " ", filtered_text)
        if not filtered_text.strip():
            continue
        
        text_boxes.append((left, filtered_text.strip()))
        confidences.append(conf / 100.0)  # 0-1 스케일로 변환
    
    # x 좌표 기준 정렬
    sorted_items = sorted(zip(text_boxes, confidences), key=lambda x: x[0][0])
    if not sorted_items:
        return [], []
    
    text_boxes, confidences = zip(*sorted_items)
    
    all_tokens = []
    token_confidences = []
    for (_, t), c in zip(text_boxes, confidences):
        tokens = t.split()
        all_tokens.extend(tokens)
        token_confidences.extend([c] * len(tokens))
    
    return all_tokens, token_confidences

def extract_text_from_easyocr_result(
        result,
        min_confidence=0.4, # EasyOCR uses 0-1 scale
        allowed_chars="가-힣0-9",
        debug=False,
        only_horizontal=True
    ):
    """
    EasyOCR 결과에서 텍스트와 신뢰도를 추출
    """
    text_boxes = []
    confidences = []
    
    for detection in result:
        bbox, text, conf = detection
        
        # 빈 텍스트나 신뢰도가 없는 경우 건너뛰기
        if not text.strip() or conf < min_confidence:
            continue
            
        # 허용된 문자만 필터링
        filtered_text = re.sub(f"[^{allowed_chars}]", " ", text)
        filtered_text = re.sub(" +", " ", filtered_text)
        if not filtered_text.strip():
            continue
        
        # bbox의 왼쪽 x좌표 사용
        left = bbox[0][0]
        text_boxes.append((left, filtered_text.strip()))
        confidences.append(conf)
    
    # x 좌표 기준 정렬
    sorted_items = sorted(zip(text_boxes, confidences), key=lambda x: x[0][0])
    if not sorted_items:
        return [], []
    
    text_boxes, confidences = zip(*sorted_items)
    
    all_tokens = []
    token_confidences = []
    for (_, t), c in zip(text_boxes, confidences):
        tokens = t.split()
        all_tokens.extend(tokens)
        token_confidences.extend([c] * len(tokens))
    
    return all_tokens, token_confidences

########################################
# 배경 제거를 위한 전처리
########################################

def remove_background_borders(img):
    """
    이미지의 검은색/흰색 배경을 제거하려는 시도
    """
    if img is None:
        return None
        
    # 그레이스케일 변환
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 흰색/검은색 마스크
    black_mask = gray < 30
    white_mask = gray > 225
    background_mask = black_mask | white_mask
    
    # 모폴로지로 노이즈 제거
    kernel = np.ones((5,5), np.uint8)
    background_mask = cv2.morphologyEx(background_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel)
    
    # 전경 마스크
    foreground_mask = ~background_mask.astype(bool)
    coords = np.column_stack(np.where(foreground_mask))
    if len(coords) == 0:
        return img
    
    # 경계 상자
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    margin = int(min(img.shape[0], img.shape[1]) * 0.01)
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(img.shape[1], x_max + margin)
    y_max = min(img.shape[0], y_max + margin)
    
    return img[y_min:y_max, x_min:x_max]

########################################
# 이미지 크기를 정규화 (리사이즈)
########################################

def normalize_image_size(img, target_width=1600, min_height=900):
    """
    이미지 크기를 정규화하고 작은 경우 업스케일
    """
    h, w = img.shape[:2]
    
    if h < min_height or w < target_width:
        scale = max(target_width / w, min_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    if w > target_width:
        scale = target_width / w
        new_h = int(h * scale)
        return cv2.resize(img, (target_width, new_h))
    
    return img

########################################
# 여러 변환을 모두 시도하여, OCR 결과가 "가장 좋게" 나오는 것을 선택
########################################

def get_transformations(img):
    """
    8가지 대표 변환 생성 후 반환.
    원하는 만큼 늘리거나 줄일 수 있음.
    """
    transformations = {}
    
    # 0) 원본
    transformations["none"] = img
    
    # 1) rot90(시계방향 90도)
    transformations["rot90"] = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    # 2) rot180
    transformations["rot180"] = cv2.rotate(img, cv2.ROTATE_180)
    
    # 3) rot270
    transformations["rot270"] = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # 4) hflip (수평 뒤집기)
    transformations["hflip"] = cv2.flip(img, 1)
    
    # 5) vflip (수직 뒤집기)
    transformations["vflip"] = cv2.flip(img, 0)
    
    # 6) rot90 후 hflip
    rot90_hf = cv2.flip(transformations["rot90"], 1)
    transformations["rot90_hflip"] = rot90_hf
    
    # 7) rot90 후 vflip
    rot90_vf = cv2.flip(transformations["rot90"], 0)
    transformations["rot90_vflip"] = rot90_vf
    
    return transformations

def process_image(
    img_path, 
    debug=False,
    valid_words=None
):
    """
    EasyOCR을 사용하도록 수정된 이미지 처리 함수
    """
    # EasyOCR reader 초기화 (첫 실행시에만)
    if not hasattr(process_image, 'reader'):
        process_image.reader = easyocr.Reader(['ko', 'en'])
    
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise ValueError(f"Failed to read image: {img_path}")
    
    # 이미지 전처리 부분은 동일
    original_img = normalize_image_size(original_img)
    sharpening_kernel = np.array([[-1,-1,-1],
                                 [-1, 9,-1],
                                 [-1,-1,-1]])
    original_img = cv2.filter2D(original_img, -1, sharpening_kernel)
    original_img = remove_background_borders(original_img)
    
    # 변환 후보 생성
    candidates = get_transformations(original_img)
    
    transform_scores = {}
    transform_texts = {}
    
    if debug:
        print("\n=== OCR Results for Each Transformation ===")
        print(f"Image: {os.path.basename(img_path)}\n")
    
    # tqdm으로 변환 진행상황 표시
    for name, candidate_img in tqdm(candidates.items(), desc="Trying transformations", leave=False):
        try:
            result = process_image.reader.readtext(candidate_img)
            tokens, confidences = extract_text_from_easyocr_result(
                result,
                min_confidence=0.4,
                allowed_chars="가-힣0-9",
                debug=debug
            )
            
            joined_text = " ".join(tokens)
            
            if tokens:
                avg_conf = sum(confidences) / len(confidences)
                score_simple = len(tokens) * avg_conf
            else:
                score_simple = 0
            
            eq_score = evaluate_text_quality(joined_text, valid_words=valid_words)
            final_score = score_simple + eq_score
            
            transform_scores[name] = final_score
            transform_texts[name] = joined_text
            
            # debug가 True일 때만 로그 출력
            if debug:
                print(f"[{name}]")
                print(f"- Text: {joined_text}")
                print(f"- Tokens: {len(tokens)}")
                print(f"- Avg Confidence: {score_simple:.3f}")
                print(f"- Quality Score: {eq_score:.1f}")
                print(f"- Final Score: {final_score:.1f}\n")
            
        except Exception as e:
            if debug:
                print(f"Error on transform '{name}': {str(e)}")
            transform_scores[name] = 0
            transform_texts[name] = ""
    
    best_transform = max(transform_scores, key=transform_scores.get)
    best_score = transform_scores[best_transform]
    best_text = transform_texts[best_transform]
    
    if debug:
        print("\n=== Best Result ===")
        print(f"Transform: {best_transform}")
        print(f"Score: {best_score:.1f}")
        print(f"Text: {best_text}")
    
    final_img = candidates[best_transform]
    return final_img, best_transform, best_text


def display_results(img_path, debug=False, valid_words=None, ocr=None):
    """
    화면에 원본 vs 최종 복원 이미지를 보여주는 시연용 함수
    """
    final_img, transform_type, best_text = process_image(img_path, debug=debug, valid_words=valid_words)
    
    original = cv2.imread(img_path)
    if original is None:
        print(f"Failed to read image: {img_path}")
        return

    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.imshow(original_rgb)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(final_rgb)
    plt.title(f'Processed\nTransform: {transform_type}')
    plt.axis('off')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # 주피터 노트북 환경에서 이미지를 표시
    try:
        from IPython.display import display, Image
        display(Image(data=buf.getvalue()))
    except ImportError:
        print("IPython display not available")
    plt.close()
    
    print("\nBest OCR Result:")
    print(f"Transformation: {transform_type}")
    print(f"Text: {best_text}")


def save_processed_image(img_path, output_dir='data/processed/test_v2', debug=False, valid_words=None):
    """
    단일 이미지를 처리 & 결과 저장
    """
    os.makedirs(output_dir, exist_ok=True)
    
    final_img, transform_type, best_text = process_image(img_path, debug=debug, valid_words=valid_words)
    
    base_name = os.path.basename(img_path)
    output_path = os.path.join(output_dir, base_name)
    
    cv2.imwrite(output_path, final_img)
    
    if debug:
        print(f"[save_processed_image] Saved: {output_path}")
        print(f"Transform: {transform_type}")
        print(f"OCR Text: {best_text}")
    
    return output_path


def process_directory(input_dir, output_dir, debug=False, valid_words=None):
    """
    디렉토리 내 모든 이미지를 변환/복원하여 저장
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
        image_files.extend(glob.glob(os.path.join(input_dir, '**', ext.upper()), recursive=True))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    # tqdm으로 전체 진행상황 표시
    pbar = tqdm(image_files, desc="Processing images")
    for img_path in pbar:
        try:
            rel_path = os.path.relpath(img_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 현재 처리 중인 파일명을 진행바에 표시
            pbar.set_description(f"Processing {os.path.basename(img_path)}")
            
            final_img, transform_type, best_text = process_image(img_path, debug=debug, valid_words=valid_words)
            cv2.imwrite(output_path, final_img)
            
        except Exception as e:
            if debug:
                print(f"\nError processing {img_path}: {str(e)}")
            continue


def display_multiple_results(input_dir, start_idx=0, n=10):
    """
    입력 디렉토리에서 start_idx부터 n개 이미지의 원본과 복원 결과를 출력
    
    Args:
        input_dir (str): 입력 디렉토리 경로
        start_idx (int): 작할 이미지 인덱스 (0-based)
        n (int): 표시할 이미지 개수
    """
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
        image_files.extend(glob.glob(os.path.join(input_dir, '**', ext.upper()), recursive=True))
    
    image_files = sorted(image_files)
    total_images = len(image_files)
    
    if start_idx >= total_images:
        print(f"Start index {start_idx} is out of range. Total images: {total_images}")
        return
    
    # 시작 인덱스부터 n개 선택
    selected_files = image_files[start_idx:start_idx + n]
    
    for i, img_path in enumerate(selected_files, start=start_idx):
        print(f"\nProcessing image {i} of {total_images}: {os.path.basename(img_path)}")
        display_results(img_path, debug=True)


if __name__ == "__main__":
    # 예시: 특정 디렉토리 내 이미지를 처리
    # "use_angle_cls=True" + "only_horizontal=True"로
    # 수평(0/180) 텍스트만 OCR에 최종 반영
    process_directory('data/processed/restore_angle_image', 'data/processed/restore_angle_2', debug=False)