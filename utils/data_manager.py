from pathlib import Path
import json
import shutil
from PIL import Image
from utils.temp_manager import TempManager
import os

class DataManager:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if DataManager._instance is not None:
            raise Exception("DataManager는 싱글톤 클래스입니다.")
        
        self.data_path = Path(__file__).parent / 'data.json'
        self.img_path = Path(__file__).parents[1] / 'img'
        self._data = self._load_data()
        DataManager._instance = self
    
    def _load_data(self):
        """데이터 로드"""
        default_data = {
            'macro_list': {},
            'settings_main': {
                'resolution': None,
                'custom': False
            }
        }
        
        try:
            if self.data_path.exists():
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 기존 매크로에 program 필드 추가
                    for macro in data.get('macro_list', {}).values():
                        if 'program' not in macro:
                            macro['program'] = None
                    return data
            return default_data
        except Exception as e:
            print(f"데이터 로드 중 오류 발생: {e}")
            return default_data
    
    def save_data(self):
        """데이터 저장"""
        try:
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"데이터 저장 중 오류 발생: {e}")
            return False
    
    def _wizard_actions_common(self, macro_key, mapping, starting_action_number):
        """
        공통 로직: TempManager의 painted 이미지 복사 및 액션 데이터 생성

        Args:
            macro_key (str): 매크로 키.
            mapping (dict): {temp_label: (액션 이름, 파일 이름)} 형식의 매핑.
            starting_action_number (int): 액션 번호를 시작할 번호.
        
        Returns:
            dict: 생성된 액션 데이터 딕셔너리.
        """
        temp_manager = TempManager.get_instance()

        macro_name = self._data['macro_list'][macro_key]['name']
        macro_folder = self.img_path / macro_name
        macro_folder.mkdir(exist_ok=True)

        # temp_manager에서 원본 드래그 영역 좌표들을 가져옵니다.
        original_drag_areas = temp_manager.get_original_drag_areas()

        actions = {}
        action_number = starting_action_number

        for temp_label, (action_name, filename) in mapping.items():
            # temp_label에 따라 적절한 이미지 경로 가져오기
            if temp_label == 'step1':
                temp_image_path = temp_manager.get_temp_image(1)
                
                # 원본 이미지에도 전체 이미지 크기를 좌표로 저장 (스케일링 계산용)
                if temp_image_path and os.path.exists(temp_image_path):
                    try:
                        # 이미지 크기 가져오기
                        import cv2
                        img = cv2.imread(str(temp_image_path))
                        if img is not None:
                            height, width = img.shape[:2]
                            # 전체 이미지 크기를 템플릿 좌표로 설정 (x=0, y=0, width, height)
                            coordinates = [0, 0, width, height]
                            print(f"원본 템플릿 이미지 크기: {width}x{height}, 좌표: {coordinates}")
                        else:
                            coordinates = None
                            print(f"원본 템플릿 이미지를 읽을 수 없음: {temp_image_path}")
                    except Exception as e:
                        coordinates = None
                        print(f"원본 템플릿 이미지 크기 읽기 실패: {e}")
                else:
                    coordinates = None
            else:
                temp_image_path = temp_manager.get_painted_image(temp_label)
                # 드래그 영역 좌표 정보 획득
                drag_area = original_drag_areas.get(temp_label)
                coordinates = ([drag_area['x'], drag_area['y'], 
                              drag_area['width'], drag_area['height']]
                                 if drag_area else None)

            # 이미지 복사 및 액션 데이터 생성
            if temp_image_path and os.path.exists(temp_image_path):
                shutil.copy2(temp_image_path, macro_folder / filename)
            
            action_key = f"A{action_number}"
            actions[action_key] = {
                'name': action_name,
                'type': 'image',
                'number': action_number,
                'image': str(macro_folder / filename),
                'priority': False,
                'coordinates': coordinates
            }
            action_number += 1

        return actions

    def create_wizard_actions(self, macro_key):
        """
        신규 매크로 생성 시, 초기 액션 생성 및 이미지 복사.
        """
        # 신규 매크로에서는 고정된 매핑과 번호를 사용
        mapping = {
            'step1': ('원본 이미지', 'A1.png'),
            'minus': ('- 버튼 이미지', 'A2.png'),
            'plus': ('+ 버튼 이미지', 'A3.png'),
            'time': ('예상시간 이미지', 'A4.png'),
            'reject': ('거부버튼 이미지', 'A5.png'),
            'accept': ('접수버튼 이미지', 'A6.png')
        }
        # 액션 번호는 1부터 시작 (A1 → number 1 등)
        actions = self._wizard_actions_common(macro_key, mapping, starting_action_number=1)
        self._data['macro_list'][macro_key]['actions'] = actions
        return self.save_data()

    def add_wizard_actions(self, macro_key):
        """
        기존 매크로에 대해서 추가 액션 생성 및 이미지 복사.
        """
        actions = self._data['macro_list'][macro_key]['actions']
        macro_name = self._data['macro_list'][macro_key]['name']
        macro_folder = self.img_path / macro_name

        # 기존 액션에서 가장 큰 번호 산출 (없으면 0)
        last_action_number = max([action['number'] for action in actions.values()], default=0)
        
        # 추가 액션은 기존 번호 + 1부터 할당
        mapping = {
            'step1': ('원본 이미지', f'A{last_action_number + 1}.png'),
            'minus': ('- 버튼 이미지', f'A{last_action_number + 2}.png'),
            'plus': ('+ 버튼 이미지', f'A{last_action_number + 3}.png'),
            'time': ('예상시간 이미지', f'A{last_action_number + 4}.png'),
            'reject': ('거부버튼 이미지', f'A{last_action_number + 5}.png'),
            'accept': ('접수버튼 이미지', f'A{last_action_number + 6}.png')
        }
        new_actions = self._wizard_actions_common(macro_key, mapping, starting_action_number=last_action_number+1)
        actions.update(new_actions)
        return self.save_data()
    
    # def save_cropped_images(self, macro_key, original_image, drag_areas):
    #     """드래그 영역으로 이미지 크롭하여 저장"""
    #     macro_name = self._data['macro_list'][macro_key]['name']
    #     macro_folder = self.img_path / macro_name
    #     macro_folder.mkdir(exist_ok=True)
        
    #     # PIL Image로 변환
    #     img = Image.open(original_image)
        
    #     # 각 영역별 이미지 크롭 및 저장
    #     area_files = {
    #         'minus': 'A0.png',
    #         'plus': 'A1.png',
    #         'time': 'A2.png',
    #         'reject': 'A3.png',  # 거절버튼
    #         'accept': 'A4.png'   # 수락버튼
    #     }
        
    #     for area_name, filename in area_files.items():
    #         if area_name in drag_areas and drag_areas[area_name]:
    #             rect = drag_areas[area_name]
    #             cropped = img.crop((rect.x(), rect.y(), 
    #                               rect.x() + rect.width(), 
    #                               rect.y() + rect.height()))
    #             cropped.save(macro_folder / filename)
                
    #     return True
    
    def create_delay_action(self, macro_key, delay_time):
        """딜레이 액션 생성"""
        try:
            actions = self._data['macro_list'][macro_key]['actions']
            # 기존 액션에서 가장 큰 번호 산출 (없으면 0)
            last_action_number = max([action['number'] for action in actions.values()], default=0)
            new_number = last_action_number + 1
            new_key = f'A{new_number}'
            
            # 딜레이 액션 생성
            actions[new_key] = {
                'name': f'딜레이: {delay_time}초',
                'type': 'delay',
                'number': new_number,
                'value': float(delay_time),
                'priority': False
            }
            
            return self.save_data()
        except Exception as e:
            print(f"딜레이 액션 생성 중 오류 발생: {e}")
            return False
    
    # def create_image_action(self, macro_key, name, image_path):
    #     """이미지 액션 생성"""
    #     try:
    #         actions = self._data['macro_list'][macro_key]['actions']
            
    #         # 새로운 액션 키 생성 (마지막 번호 + 1)
    #         last_num = -1
    #         for key in actions.keys():
    #             if key.startswith('A'):
    #                 try:
    #                     num = int(key[1:])
    #                     last_num = max(last_num, num)
    #                 except ValueError:
    #                     continue
    #         new_key = f'A{last_num + 1}'
            
    #         # 새로운 number 계산 (현재 가장 큰 number + 1)
    #         new_number = max([action['number'] for action in actions.values()], default=0) + 1
            
    #         # 새로운 액션 생성
    #         actions[new_key] = {
    #             'name': name,
    #             'type': 'image',
    #             'number': new_number,
    #             'image': str(image_path),
    #             'priority': False
    #         }
            
    #         return new_key
            
    #     except Exception as e:
    #         print(f"이미지 액션 생성 중 오류 발생: {e}")
    #         return None
    
    def copy_macro(self, original_macro_key, new_name):
        """매크로 복제
        
        Args:
            original_macro_key (str): 원본 매크로 키
            new_name (str): 새로운 매크로 이름
            
        Returns:
            str: 새로운 매크로 키
        """
        try:
            # 새로운 매크로 키 생성 (수정된 부분)
            macro_keys = self._data['macro_list'].keys()
            next_num = 1
            while f"M{next_num}" in macro_keys:
                next_num += 1
            new_macro_key = f"M{next_num}"
            
            # 원본 매크로 데이터 복사
            original_macro = self._data['macro_list'][original_macro_key]
            new_macro = {
                'name': new_name,
                'program': original_macro.get('program'),
                'actions': {}
            }
            
            # 액션 데이터 복사
            for action_key, action_data in original_macro['actions'].items():
                new_macro['actions'][action_key] = action_data.copy()
            
            # 이미지 파일 복사
            original_folder = self.img_path / original_macro['name']
            new_folder = self.img_path / new_name
            if original_folder.exists():
                # 기존 폴더가 있다면 삭제
                if new_folder.exists():
                    shutil.rmtree(new_folder)
                # 폴더 복사
                shutil.copytree(original_folder, new_folder)
                
                # 이미지 경로 업데이트
                for action in new_macro['actions'].values():
                    if action['type'] == 'image':
                        old_path = Path(action['image'])
                        action['image'] = str(new_folder / old_path.name)
            
            # 새로운 매크로 저장
            self._data['macro_list'][new_macro_key] = new_macro
            self.save_data()
            
            return new_macro_key
            
        except Exception as e:
            print(f"매크로 복제 중 오류 발생: {e}")
            return None