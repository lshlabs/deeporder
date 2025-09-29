import Quartz

def print_window_names():
    """현재 실행 중인 모든 창 정보 출력"""
    # 모든 윈도우 정보 가져오기
    window_list = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID
    )
    
    print("\n=== 실행 중인 창 목록 ===")
    for window in window_list:
        owner = window.get(Quartz.kCGWindowOwnerName, '')
        name = window.get(Quartz.kCGWindowName, '')
        
        if owner or name:
            print(f"앱: {owner}")
            if name:
                print(f"창 이름: {name}")
            print("---")

if __name__ == "__main__":
    print_window_names()