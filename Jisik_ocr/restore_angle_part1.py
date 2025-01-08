import os
os.chdir("/data/ephemeral/home/upstage-cv")

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def is_nearly_horizontal_or_vertical(angle, threshold=0.1):
    """각도가 수평(0도 또는 180도) 또는 수직(90도 또는 270도)에 너무 가까운지 확인"""
    angle = angle % 360
    return any(abs((angle - target) % 180) <= threshold for target in [0, 90])

def rotate_by_contour_angle(image):
    """
    1) 그레이스케일/블러/이진화/모폴로지
    2) 가장 큰 윤곽선에 대해 점진적 근사화 (epsilon 증가)
    3) 꼭짓점 수가 4/6/8개라면 각도 계산 후 회전
    4) 아니면 원본+윤곽선 그대로 반환
    """
    # --- (1) 전처리: 그레이스케일, 블러, 이진화, 모폴로지 ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 배경을 검정(0), 물체(문서)를 흰색(255)로
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # 윤곽선이 없으면 (회전 불가 → 원본 반환)
        contour_image = image.copy()  # 윤곽선 표시 이미지
        return image, contour_image, False  # success=False

    # 가장 큰 윤곽선
    largest_contour = max(contours, key=cv2.contourArea)

    # --- (2) 점진적 근사화: 4/6/8각형인지 확인 ---
    epsilon_start = 0.01
    epsilon_max   = 0.2
    epsilon_step  = 0.01
    
    current_epsilon = epsilon_start
    found_polygon = False
    final_approx  = None
    final_vertices = None

    while current_epsilon <= epsilon_max:
        approx = cv2.approxPolyDP(
            largest_contour, 
            current_epsilon * cv2.arcLength(largest_contour, True), 
            True
        )
        vertices = len(approx)
        
        if vertices in [4, 6, 8]:
            found_polygon = True
            final_approx = approx
            final_vertices = vertices
            print(f"[rotate_by_contour_angle] Found {vertices}-gon with epsilon={current_epsilon:.3f}")
            break
        
        current_epsilon += epsilon_step

    contour_image = image.copy()
    cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)

    if not found_polygon:
        print("[rotate_by_contour_angle] Warning: Could not approximate to 4,6,8각형. Returning original.")
        return image, contour_image, False  # success=False

    # --- (3) 각도 계산 + 회전 ---
    # final_approx(4,6,8 각형) 기준으로 에지 각도를 구해 “수평에 가장 가까운 각도”를 찾아 회전
    angles = []
    for i in range(final_vertices):
        pt1 = final_approx[i][0]
        pt2 = final_approx[(i + 1) % final_vertices][0]
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        side_angle = np.degrees(np.arctan2(dy, dx)) % 180
        angles.append(side_angle)

    # 수평/수직과 거의 일치(±0.1도)하는 각도는 회전 대상에서 제외
    horizontal_diff = []
    valid_angles = []
    for angle in angles:
        if not is_nearly_horizontal_or_vertical(angle, threshold=0.1):
            # 수평(0/180)과의 차이 절댓값
            diff = min(abs(angle % 180), abs((angle - 180) % 180))
            horizontal_diff.append(diff)
            valid_angles.append(angle)
    
    if valid_angles:
        closest_to_horizontal = valid_angles[horizontal_diff.index(min(horizontal_diff))]
    else:
        closest_to_horizontal = 0  # 모두 수평/수직이면 회전 불필요
    
    rotation_angle = -closest_to_horizontal
    print(f"[rotate_by_contour_angle] Selected rotation angle: {rotation_angle:.3f} deg")

    # --- (4) 실제 회전 ---
    (img_h, img_w) = image.shape[:2]
    angle_rad = np.deg2rad(rotation_angle)
    
    # 회전 후 필요한 새 이미지 크기
    cos = abs(np.cos(angle_rad))
    sin = abs(np.sin(angle_rad))
    new_w = int(img_h * sin + img_w * cos)
    new_h = int(img_h * cos + img_w * sin)
    
    # 회전 행렬
    M = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), -rotation_angle, 1.0)
    
    # 새 캔버스 중심 보정
    M[0, 2] += (new_w - img_w) / 2
    M[1, 2] += (new_h - img_h) / 2
    
    rotated = cv2.warpAffine(
        image, 
        M, 
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)  # 흰색 배경
    )
    
    return rotated, contour_image, True  # success=True


def display_images(index, original, contour_image, rotated, 
                   title1="Original", title2="Contour", title3="Restored"):
    """
    3장 비교 시각화
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    fig.suptitle(f"Image Index: {index}", fontsize=16)
    
    ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax1.set_title(title1)
    ax1.axis('off')
    
    ax2.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    ax2.set_title(title2)
    ax2.axis('off')
    
    ax3.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    ax3.set_title(title3)
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    input_folder = "data/raw/test"
    output_folder = "data/processed/restore_angle_image"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [
        f for f in os.listdir(input_folder) 
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    
    non_polygon_count = 0  # 4,6,8각형이 아닌 (또는 근사화 실패) 이미지 카운트
    
    for idx, file_name in enumerate(image_files):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        img = cv2.imread(input_path)
        if img is None:
            print(f"Cannot read image: {input_path}")
            continue

        # --- 단일 함수로 회전/윤곽/성공여부 반환 ---
        rotated_img, contour_img, success = rotate_by_contour_angle(img)
        
        if success:
            print(f"[MAIN] Rotation success. Saving rotated image: {output_path}")
            cv2.imwrite(output_path, rotated_img)
            # 필요시 시각화
            # display_images(idx, img, contour_img, rotated_img)
        else:
            # 4,6,8각형이 아니거나 윤곽선이 없어서 회전 실패 → 원본을 그대로 저장
            non_polygon_count += 1
            print(f"[MAIN] Not a valid polygon or no contour. Saving original: {output_path}")
            cv2.imwrite(output_path, img)
            # 필요시 시각화
            # display_images(idx, img, contour_img, img)

    print(f"\n총 이미지 수: {len(image_files)}")
    print(f"4,6,8각형으로 근사화되지 않은 이미지 수: {non_polygon_count}")