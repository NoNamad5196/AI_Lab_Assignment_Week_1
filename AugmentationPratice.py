import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # 이미지를 불러오고 NumPy 배열로 변환하기 위해 사용합니다.

# --- 1. 이미지 불러오기 ---
def load_image(image_path):
    """이미지 파일을 불러와 NumPy 배열로 변환합니다."""
    # 'L' 모드는 이미지를 흑백(grayscale)으로 변환합니다. 
    return np.array(Image.open(image_path).convert('L'))

# --- 2. 이미지 증강 함수 구현 ---

def horizontal_flip(image):
    """이미지를 좌우로 반전시킵니다."""
    # NumPy의 슬라이싱 기능을 사용하여 배열의 열 순서를 뒤집습니다.
    return image[:, ::-1]

def adjust_brightness(image, factor):
    """이미지의 밝기를 조절합니다."""
    # 이미지의 모든 픽셀 값에 factor를 곱합니다.
    # np.clip 함수는 배열의 값이 특정 범위(0~255)를 벗어나지 않도록 잘라줍니다.
    adjusted_image = image * factor
    return np.clip(adjusted_image, 0, 255).astype(np.uint8)

def rotate_image(image, angle_degrees):
    """이미지를 지정된 각도만큼 회전시킵니다. (중심점 기준)"""
    # 각도를 라디안으로 변환
    angle_rad = np.deg2rad(angle_degrees)
    
    # 이미지의 높이와 너비, 중심점 좌표
    height, width = image.shape
    center_y, center_x = height // 2, width // 2

    # 회전 후 결과 이미지를 담을 빈 배열 생성 (원본과 동일한 크기)
    rotated_image = np.zeros_like(image)

    # 결과 이미지의 모든 픽셀 위치 (x', y')를 순회
    for y_prime in range(height):
        for x_prime in range(width):
            # 역회전 변환을 사용하여 원본 이미지의 좌표 (x, y)를 계산
            # (x', y')에서 중심을 빼고, 회전 행렬의 역행렬을 곱한 후, 다시 중심을 더함
            x = (x_prime - center_x) * np.cos(angle_rad) + (y_prime - center_y) * np.sin(angle_rad) + center_x
            y = -(x_prime - center_x) * np.sin(angle_rad) + (y_prime - center_y) * np.cos(angle_rad) + center_y

            # 계산된 (x, y)가 원본 이미지 범위 안에 있는지 확인
            if 0 <= x < width and 0 <= y < height:
                # 가장 가까운 픽셀 값을 가져오는 '최근방 이웃 보간법' 사용
                rotated_image[y_prime, x_prime] = image[int(y), int(x)]
                
    return rotated_image

# --- 3. 결과 시각화 ---
def visualize_results(original, flipped, bright, dark, rotated):
    """원본과 변환된 이미지들을 함께 보여줍니다."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(flipped, cmap='gray')
    axes[1].set_title("Flipped")
    axes[1].axis('off')

    axes[2].imshow(bright, cmap='gray')
    axes[2].set_title("Brighter (x1.5)")
    axes[2].axis('off')

    axes[3].imshow(dark, cmap='gray')
    axes[3].set_title("Darker (x0.5)")
    axes[3].axis('off')

    axes[4].imshow(rotated, cmap='gray')
    axes[4].set_title("Rotated (30°)")
    axes[4].axis('off')

    plt.savefig("result.png")
    print("결과가 result.png 파일로 저장되었습니다.") # 확인 메시지 추가

# --- 4. 프로그램 실행 ---
if __name__ == "__main__":
    # 여기에 자신의 이미지 파일 경로를 입력하세요.
    try:
        image_path = "my_image.png"
        
        original_image = load_image(image_path)
        
        # 각 증강 함수를 실행
        flipped_image = horizontal_flip(original_image)
        bright_image = adjust_brightness(original_image, 1.5)
        dark_image = adjust_brightness(original_image, 0.5)
        rotated_image = rotate_image(original_image, 30) # 30도 회전
        
        # 결과 시각화
        visualize_results(original_image, flipped_image, bright_image, dark_image, rotated_image)

    except FileNotFoundError:
        print(f"오류: '{image_path}' 파일을 찾을 수 없습니다.")
        print("코드의 image_path 변수에 올바른 이미지 파일 경로를 입력했는지 확인해주세요.")