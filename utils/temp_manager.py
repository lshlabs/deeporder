from pathlib import Path
import json
import shutil
from PIL import Image

class TempManager:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if TempManager._instance is not None:
            raise Exception("TempManager는 싱글톤 클래스입니다.")
            
        self.temp_dir = Path(__file__).parents[1] / 'temp'
        self.temp_dir.mkdir(exist_ok=True)
        self.temp_path = self.temp_dir / 'tempdata.json'
        self._temp_data = self._load_temp_data()
        TempManager._instance = self
        
    def _load_temp_data(self):
        """임시 데이터 로드"""
        default_data = {
            'drag_areas': {         # GUI 상에서의(스케일된) 드래그 영역 좌표
                'plus': None,
                'minus': None,
                'time': None,
                'accept': None,
                'reject': None
            },
            'original_drag_areas': {  # 원본 이미지 좌표로 계산된 드래그 영역
                'plus': None,
                'minus': None,
                'time': None,
                'accept': None,
                'reject': None
            },
            'temp_images': {
                'step1': None,
                'step2': None,
                'step3': None
            },
            'painted_images': {
                'plus': None,
                'minus': None,
                'time': None,
                'accept': None,
                'reject': None
            }
        }
        
        try:
            if self.temp_path.exists():
                with open(self.temp_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 기본 데이터에 없는 키가 있다면 추가합니다.
                for key, value in default_data.items():
                    if key not in data:
                        data[key] = value
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if sub_key not in data[key]:
                                data[key][sub_key] = sub_value
                return data
            return default_data
        except Exception as e:
            print(f"임시 데이터 로드 중 오류 발생: {e}")
            return default_data
            
    def save_temp_data(self):
        """임시 데이터 저장"""
        try:
            with open(self.temp_path, 'w', encoding='utf-8') as f:
                json.dump(self._temp_data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"임시 데이터 저장 중 오류 발생: {e}")
            return False
            
    def save_drag_areas(self, drag_areas):
        """드래그 영역 저장 (스케일된 좌표)"""
        current_areas = self._temp_data['drag_areas']
        for label, rect in drag_areas.items():
            current_areas[label] = {
                'x': rect.x(),
                'y': rect.y(),
                'width': rect.width(),
                'height': rect.height()
            } if rect else None
        return self.save_temp_data()
        
    def get_drag_areas(self):
        """드래그 영역 가져오기 (스케일된 좌표)"""
        return self._temp_data['drag_areas']
        
    def save_original_drag_areas(self, original_drag_areas):
        """원본 이미지 좌표로 계산된 드래그 영역 저장"""
        current_orig = self._temp_data['original_drag_areas']
        for label, rect in original_drag_areas.items():
            current_orig[label] = {
                'x': rect.x(),
                'y': rect.y(),
                'width': rect.width(),
                'height': rect.height()
            } if rect else None
        return self.save_temp_data()
        
    def get_original_drag_areas(self):
        """원본 이미지 좌표로 계산된 드래그 영역 가져오기"""
        return self._temp_data['original_drag_areas']
        
    def save_temp_image(self, image, step):
        """수정된 이미지 저장"""
        image_path = self.temp_dir / f'temp_step{step}.png'
        image.save(str(image_path))
        self._temp_data['temp_images'][f'step{step}'] = str(image_path)
        return self.save_temp_data()
        
    def get_temp_image(self, step):
        """수정된 이미지 가져오기"""
        return self._temp_data['temp_images'][f'step{step}']
        
    def save_painted_image(self, image, label):
        """드래그 영역이 그려진(페인팅된) 이미지 저장"""
        image_path = self.temp_dir / f'painted_{label}.png'
        image.save(str(image_path))
        self._temp_data['painted_images'][label] = str(image_path)
        return self.save_temp_data()
        
    def get_painted_image(self, label):
        """페인팅된 이미지 가져오기"""
        return self._temp_data['painted_images'][label]
        
    def clear_temp_data(self):
        """임시 데이터 초기화"""
        for path in self.temp_dir.glob('*.png'):
            path.unlink(missing_ok=True)
        if self.temp_path.exists():
            self.temp_path.unlink()
        self._temp_data = self._load_temp_data()