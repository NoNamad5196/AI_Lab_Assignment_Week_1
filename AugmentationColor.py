import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- 1. 이미지 불러오기 (컬러) ---
def load_image(image_path):
    """이미지 파일을 불러와 컬러 NumPy 배열로 변환합니다."""
    # .convert('L')을 제거하여 원본 컬러를 그대로 사용합니다.
    return np.array(Image.open(image_path))

# --- 2. 이미지 증강 함수 구현 (컬러 기준) ---

def horizontal_flip(image):
    """이미지를 좌우로 반전시킵니다."""
    # 컬러(3D) 배열의 좌우를 뒤집습니다.
    return image[:, ::-1, :]

def adjust_brightness(image, factor):
    """이미지의 밝기를 조절합니다."""
    # 각 채널(R,G,B)의 값을 조절하고 범위를 0~255로 유지합니다.
    adjusted_image = image.astype(float) * factor
    return np.clip(adjusted_image, 0, 255).astype(np.uint8)

def rotate_image(image, angle_degrees):
    """컬러 이미지를 지정된 각도만큼 회전시킵니다."""
    angle_rad = np.deg2rad(angle_degrees)
    
    # 컬러 이미지의 높이, 너비, 채널 정보를 가져옵니다.
    height, width, channels = image.shape
    center_y, center_x = height // 2, width // 2

    # 원본과 동일한 3차원 형태의 빈 배열을 생성합니다.
    rotated_image = np.zeros_like(image)

    # 결과 이미지의 모든 픽셀을 순회
    for y_prime in range(height):
        for x_prime in range(width):
            # 역회전 변환으로 원본 좌표를 계산
            x = (x_prime - center_x) * np.cos(angle_rad) + (y_prime - center_y) * np.sin(angle_rad) + center_x
            y = -(x_prime - center_x) * np.sin(angle_rad) + (y_prime - center_y) * np.cos(angle_rad) + center_y

            # 계산된 좌표가 이미지 범위 안에 있을 경우
            if 0 <= x < width and 0 <= y < height:
                # 해당 좌표의 픽셀 값(R, G, B 채널 전체)을 복사합니다.
                rotated_image[y_prime, x_prime, :] = image[int(y), int(x), :]
                
    return rotated_image

# --- 3. 결과 시각화 및 저장 ---
def visualize_results(original, flipped, bright, dark, rotated):
    """변환 결과들을 시각화하고 하나의 이미지 파일로 저장합니다."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    images = [original, flipped, bright, dark, rotated]
    titles = ["Original", "Flipped", "Brighter (x1.5)", "Darker (x0.5)", "Rotated (30°)"]
    
    for ax, img, title in zip(axes, images, titles):
        # 컬러 이미지를 표시하므로 cmap='gray' 옵션을 사용하지 않습니다.
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    # 최종 결과를 'resultColor.png' 파일로 저장합니다.
    plt.savefig("resultColor.png")
    print("컬러 이미지 증강 결과가 resultColor.png 파일로 저장되었습니다.")

# --- 4. 프로그램 실행 ---
if __name__ == "__main__":
    try:
        image_path = "my_image.png"
        
        # 컬러 이미지를 불러옵니다.
        original_image = load_image(image_path)
        
        # 각 증강 함수를 실행합니다.
        flipped_image = horizontal_flip(original_image)
        bright_image = adjust_brightness(original_image, 1.5)
        dark_image = adjust_brightness(original_image, 0.5)
        rotated_image = rotate_image(original_image, 30) # 30도 회전
        
        # 결과를 시각화하고 저장합니다.
        visualize_results(original_image, flipped_image, bright_image, dark_image, rotated_image)

    except FileNotFoundError:
        print(f"오류: '{image_path}' 파일을 찾을 수 없습니다.")
        print("코드의 image_path 변수에 올바른 이미지 파일 경로를 입력했는지 확인해주세요.")