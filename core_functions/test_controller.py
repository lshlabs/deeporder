from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPixmap
import sys  # 추가
from image_matcher import ImageMatcher
from screen_monitor import ScreenMonitor
from mouse_controller import MouseController

# Qt 애플리케이션 인스턴스 생성 (가장 먼저 해야 함)
app = QApplication(sys.argv)  # 추가

# 이미지 매칭 예제
def test_image_matching():
    # 이미지 매처 생성
    matcher = ImageMatcher(threshold=0.8)
    
    # 템플릿 이미지 로드
    template = QPixmap('/Users/mac/Documents/GitHub/myGit/deeporder/example/쿠팡이츠 배달 레티나.png')
    
    # 모든 반환값을 받도록 수정 (정확한 수를 알 수 없으므로 가변적으로 처리)
    result = matcher.find_template(template)
    
    # 결과 튜플의 길이 확인
    print(f"반환된 값 개수: {len(result)}")
    print(f"반환된 값들: {result}")
    
    # 첫 번째 값은 성공 여부
    success = result[0]
    
    if success:
        # 두 번째 값은 위치
        location = result[1]
        # 세 번째 값은 유사도
        confidence = result[2]
        print(f"이미지 발견: 위치 {location}, 유사도 {confidence:.2f}")
    else:
        confidence = result[2]
        print(f"이미지를 찾지 못했습니다. 최대 유사도: {confidence:.2f}")

# ... 나머지 코드는 동일 ...

if __name__ == "__main__":
    test_image_matching()
    # test_monitoring_and_click()
    
    # 필요한 경우 이벤트 루프 실행 (GUI가 표시되지 않는 경우에는 필요 없음)
    # sys.exit(app.exec())