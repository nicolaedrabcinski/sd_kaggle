import pandas as pd
import numpy as np
from pathlib import Path
import torch

# Ваши результаты
results = {
    'Model': ['nhits', 'fedformer', 'residual_mlp', 'dlinear', 'xgboost_nn', 
              'xgboost_nn', 'dlinear', 'fedformer', 'cnn_attention', 'patchtst_ci',
              'residual_mlp', 'autoformer', 'tabnet', 'autoformer', 'cnn_attention',
              'tabnet', 'nbeats', 'nbeats', 'nlinear', 'nlinear', 'patchtst', 'nhits', 'crossformer'],
    'RMSE': [0.029737, 0.029766, 0.029853, 0.041078, 0.041102, 0.030130, 0.030662, 
             0.042128, 0.031004, 0.042617, 0.042923, 0.049284, 0.038720, 0.039365, 
             0.062726, 0.078821, 0.111354, 0.429702, 0.404733, 1.035654, 2.219481, 2.921042, 4.434728],
    'R²': [-0.0024, -0.0044, -0.0103, -0.0232, -0.0244, -0.0291, -0.0658, -0.0762, 
           -0.0897, -0.1014, -0.1172, -0.4729, -0.6996, -0.7566, -1.3859, -2.7674, 
           -13.0566, -110.9685, -184.6959, -649.4147, -2986.1936, -5173.1187, -11924.9951],
    'Dir_Acc': [0.5078, 0.5007, 0.5065, 0.5115, 0.5127, 0.4982, 0.5001, 0.4971, 
                0.5001, 0.5053, 0.5076, 0.5074, 0.5029, 0.4998, 0.5014, 0.5002, 
                0.5012, 0.4971, 0.5000, 0.4990, 0.5071, 0.5023, 0.5075],
    'Params': [65278847, 2685448, 55478088, 181513552, 56977818, 34594778, 494757120, 
               1666536, 20630280, 143851560, 108742824, 749032, 76122056, 1767944, 
               6889960, 150754988, 328942113, 649162377, 247378560, 90756776, 5924776, 
               115662821, 6055336],
    'Time_m': [1.1, 0.7, 1.1, 3.7, 0.5, 1.1, 32.6, 0.3, 0.9, 29.5, 1.1, 0.2, 1.1, 
               0.5, 0.2, 1.0, 2.4, 2.0, 18.2, 0.6, 0.2, 0.7, 0.1]
}

df_results = pd.DataFrame(results)

# Убираем дубликаты, оставляя лучшую версию каждой модели
df_best_models = df_results.loc[df_results.groupby('Model')['Dir_Acc'].idxmax()]

# Сортируем по Dir Acc (основная метрика)
df_best_models = df_best_models.sort_values('Dir_Acc', ascending=False)

print("🎯 ЛУЧШИЕ МОДЕЛИ ПО DIR ACC:")
print(df_best_models[['Model', 'Dir_Acc', 'RMSE', 'R²', 'Time_m']].head(10))

# Выбираем топ-5 моделей
top_models = df_best_models.head(5)['Model'].tolist()
print(f"\n🚀 ТОП-5 МОДЕЛЕЙ ДЛЯ SUBMISSION: {top_models}")

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import importlib
import sys

class RealModelLoader:
    def __init__(self, models_base_path, src_path):
        self.models_base_path = Path(models_base_path)
        self.src_path = Path(src_path)
        self.models = {}
        
        # Добавляем src в путь для импорта
        sys.path.append(str(self.src_path))
    
    def discover_models(self):
        """Находит все доступные модели"""
        model_folders = [f for f in self.models_base_path.iterdir() if f.is_dir()]
        available_models = []
        
        for folder in model_folders:
            model_file = folder / "best_model.pth"
            if model_file.exists():
                available_models.append(folder.name)
                print(f"✅ {folder.name}: best_model.pth найден")
            else:
                print(f"❌ {folder.name}: best_model.pth не найден")
                
        return available_models
    
    def load_model(self, model_name):
        """Загружает модель с правильной архитектурой"""
        model_path = self.models_base_path / model_name / "best_model.pth"
        
        if not model_path.exists():
            print(f"❌ Модель {model_name} не найдена")
            return None
            
        try:
            print(f"🔄 Загружаем {model_name}...")
            
            # Пробуем импортировать соответствующую архитектуру
            model_module = self._import_model_architecture(model_name)
            if model_module is None:
                print(f"❌ Не найден модуль для {model_name}")
                return None
            
            # Загружаем чекпоинт
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Создаем модель на основе архитектуры
            model = self._create_model_from_checkpoint(model_name, checkpoint, model_module)
            
            if model is not None:
                self.models[model_name] = model
                print(f"✅ {model_name} успешно загружена")
                return model
            else:
                print(f"❌ Не удалось создать модель {model_name}")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка загрузки {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _import_model_architecture(self, model_name):
        """Импортирует модуль с архитектурой модели"""
        try:
            if model_name == 'nhits':
                from src.models.nhits import NHiTS
                return NHiTS
            elif model_name == 'dlinear':
                from src.models.dlinear import DLinear
                return DLinear
            elif model_name == 'patchtst':
                from src.models.patchtst import PatchTST
                return PatchTST
            elif model_name == 'cnn_attention':
                from src.models.cnn_attention import CNNAttention
                return CNNAttention
            elif model_name == 'tabnet':
                from src.models.tabnet import TabNetModel
                return TabNetModel
            elif model_name == 'timesnet':
                from src.models.timesnet import TimesNet
                return TimesNet
            elif model_name == 'residual_mlp':
                # Проверяем есть ли residual_mlp в доступных файлах
                residual_mlp_path = self.src_path / 'models' / 'residual_mlp.py'
                if residual_mlp_path.exists():
                    spec = importlib.util.spec_from_file_location("residual_mlp", residual_mlp_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return getattr(module, 'ResidualMLP', None)
            else:
                print(f"⚠️  Архитектура для {model_name} не найдена, пробуем универсальный подход")
                return None
                
        except ImportError as e:
            print(f"❌ Ошибка импорта {model_name}: {e}")
            return None
    
    def _create_model_from_checkpoint(self, model_name, checkpoint, model_class):
        """Создает модель из чекпоинта"""
        try:
            if isinstance(checkpoint, dict):
                if 'hyper_parameters' in checkpoint:
                    # Используем гиперпараметры из чекпоинта
                    hparams = checkpoint['hyper_parameters']
                    
                    if model_class:
                        # Создаем модель с теми же параметрами
                        if 'seq_len' in hparams and 'pred_len' in hparams:
                            model = model_class(
                                seq_len=hparams['seq_len'],
                                pred_len=hparams['pred_len'],
                                **{k: v for k, v in hparams.items() if k not in ['seq_len', 'pred_len']}
                            )
                        else:
                            # Параметры по умолчанию
                            model = model_class()
                    else:
                        # Универсальная модель если класс не найден
                        model = self._create_universal_model(checkpoint)
                else:
                    model = self._create_universal_model(checkpoint)
                
                # Загружаем state_dict
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    
            elif hasattr(checkpoint, 'state_dict'):
                # Если это уже модель
                model = checkpoint
            else:
                model = self._create_universal_model(checkpoint)
            
            model.eval()
            return model
            
        except Exception as e:
            print(f"❌ Ошибка создания модели {model_name}: {e}")
            return None
    
    def _create_universal_model(self, checkpoint):
        """Создает универсальную модель когда архитектура неизвестна"""
        class UniversalModel(torch.nn.Module):
            def __init__(self, input_size=100, output_size=424, hidden_layers=[512, 256]):
                super().__init__()
                layers = []
                prev_size = input_size
                
                for hidden_size in hidden_layers:
                    layers.append(torch.nn.Linear(prev_size, hidden_size))
                    layers.append(torch.nn.ReLU())
                    layers.append(torch.nn.Dropout(0.1))
                    prev_size = hidden_size
                
                layers.append(torch.nn.Linear(prev_size, output_size))
                self.network = torch.nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        # Пытаемся определить размеры из state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            for key, param in state_dict.items():
                if 'weight' in key and len(param.shape) == 2:
                    if param.shape[0] == 424:  # выходной слой
                        input_size = param.shape[1]
                        return UniversalModel(input_size=input_size, output_size=424)
        
        return UniversalModel()

# Инициализация загрузчика
loader = RealModelLoader(
    models_base_path="/home/nicolaedrabcinski/sd_kaggle/models/checkpoints",
    src_path="/home/nicolaedrabcinski/sd_kaggle/src"
)

# Находим доступные модели
available_models = loader.discover_models()
print(f"\n🎯 Доступно моделей: {len(available_models)}")
print(f"📋 Список: {available_models}")

import warnings
warnings.filterwarnings("ignore")

class RealSubmissionGenerator:
    def __init__(self, data_path="/home/nicolaedrabcinski/sd_kaggle/data"):
        self.data_path = Path(data_path)
        self.target_columns = [f'target_{i}' for i in range(424)]
        
    def load_real_data(self):
        """Загружает реальные данные включая test.csv"""
        print("📂 Загружаем реальные данные...")
        
        # Основные данные
        self.train_df = pd.read_csv(self.data_path / "raw" / "train.csv")
        self.train_labels = pd.read_csv(self.data_path / "raw" / "train_labels.csv")
        self.target_pairs = pd.read_csv(self.data_path / "raw" / "target_pairs.csv")
        
        # ЗАГРУЖАЕМ РЕАЛЬНЫЙ TEST.CSV
        test_csv_path = self.data_path / "raw" / "test.csv"
        if test_csv_path.exists():
            self.test_df = pd.read_csv(test_csv_path)
            print(f"✅ test.csv загружен: {self.test_df.shape}")
        else:
            print("❌ test.csv не найден! Проверьте путь:")
            print(f"   Искомый путь: {test_csv_path}")
            # Покажем что есть в папке raw
            raw_files = list((self.data_path / "raw").glob("*"))
            print("   Файлы в raw/:")
            for f in raw_files:
                print(f"     - {f.name}")
            raise FileNotFoundError(f"test.csv не найден по пути: {test_csv_path}")
        
        # Проверяем enhanced данные
        enhanced_test_path = self.data_path / "processed" / "test_enhanced.csv"
        if enhanced_test_path.exists():
            self.enhanced_test = pd.read_csv(enhanced_test_path)
            print("✅ Enhanced test данные загружены")
        else:
            print("⚠️ Enhanced test данные не найдены, используем базовый test.csv")
            self.enhanced_test = None
        
        print(f"✅ Данные загружены: train={self.train_df.shape}, test={self.test_df.shape}, labels={self.train_labels.shape}")
        
    def get_test_data(self):
        """Возвращает тестовые данные (приоритет enhanced версии)"""
        if self.enhanced_test is not None and not self.enhanced_test.empty:
            print("🎯 Используем enhanced test данные")
            return self.enhanced_test.copy()
        else:
            print("🎯 Используем базовый test.csv")
            return self.test_df.copy()
    
    def prepare_real_features(self, data):
        """Подготавливает фичи на основе вашей реальной обработки"""
        # Используем те же фичи, что и при обучении
        feature_columns = []
        
        # Базовые числовые колонки (исключая date_id и targets)
        numeric_cols = [col for col in data.columns 
                       if col not in ['date_id'] + self.target_columns 
                       and pd.api.types.is_numeric_dtype(data[col])]
        
        # Если есть enhanced фичи в train, используем их как ориентир
        enhanced_train_path = self.data_path / "processed" / "train_enhanced.csv"
        if enhanced_train_path.exists():
            enhanced_train = pd.read_csv(enhanced_train_path)
            enhanced_numeric = [col for col in enhanced_train.columns 
                              if col not in ['date_id'] + self.target_columns 
                              and pd.api.types.is_numeric_dtype(enhanced_train[col])]
            
            # Выбираем общие колонки между test и enhanced train
            common_cols = list(set(numeric_cols) & set(enhanced_numeric))
            if common_cols:
                feature_columns = common_cols
                print(f"📊 Используем {len(feature_columns)} enhanced фичей")
            else:
                feature_columns = numeric_cols
                print(f"📊 Используем {len(feature_columns)} базовых фичей")
        else:
            feature_columns = numeric_cols
            print(f"📊 Используем {len(feature_columns)} базовых фичей из test.csv")
        
        return feature_columns
    
    def generate_submission_for_model(self, model, model_name, test_data):
        """Генерирует submission для конкретной модели"""
        try:
            feature_columns = self.prepare_real_features(test_data)
            
            if not feature_columns:
                print(f"❌ Нет фичей для модели {model_name}")
                print(f"   Доступные колонки в test данных: {list(test_data.columns)}")
                return None
            
            # Подготавливаем данные
            features = test_data[feature_columns].copy()
            features = features.fillna(0)  # Базовая обработка NaN
            
            print(f"🔢 Размер фичей для {model_name}: {features.shape}")
            
            # Конвертируем в tensor и делаем предсказание
            with torch.no_grad():
                input_tensor = torch.FloatTensor(features.values)
                print(f"🎯 Делаем предсказание для {len(input_tensor)} samples...")
                predictions = model(input_tensor).numpy()
            
            # Создаем submission DataFrame
            submission_df = pd.DataFrame(predictions, columns=self.target_columns)
            
            # Добавляем date_id из test данных
            if 'date_id' in test_data.columns:
                submission_df['date_id'] = test_data['date_id'].values
                print(f"📅 Добавлены date_id для {len(submission_df)} строк")
            
            print(f"✅ {model_name}: сгенерировано {len(predictions)} предсказаний, shape={predictions.shape}")
            return submission_df
            
        except Exception as e:
            print(f"❌ Ошибка генерации для {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_submission(self, submission_df, model_name, output_dir="/home/nicolaedrabcinski/sd_kaggle/submissions"):
        """Сохраняет submission файл"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Сохраняем в parquet (формат для Kaggle)
        output_path = output_dir / f"submission_{model_name}.parquet"
        submission_df.to_parquet(output_path, index=False)
        
        # Также сохраняем в CSV для проверки
        csv_path = output_dir / f"submission_{model_name}.csv"
        submission_df.to_csv(csv_path, index=False)
        
        print(f"💾 Submission сохранен: {output_path}")
        print(f"   Размер: {submission_df.shape}")
        print(f"   Колонки: {list(submission_df.columns[:3])}...")  # Покажем первые 3 колонки
        
        return output_path

# Основной скрипт генерации с реальными тестовыми данными
def generate_all_submissions_with_real_test():
    """Генерирует submission файлы для всех моделей используя реальный test.csv"""
    
    # Лучшие модели по вашим результатам
    top_models = ['xgboost_nn', 'dlinear', 'nhits', 'residual_mlp', 'patchtst_ci']
    
    print("🚀 ЗАПУСК ГЕНЕРАЦИИ SUBMISSION ФАЙЛОВ С REAL TEST.CSV")
    print(f"🎯 Целевые модели: {top_models}")
    
    # Инициализация
    loader = RealModelLoader(
        models_base_path="/home/nicolaedrabcinski/sd_kaggle/models/checkpoints",
        src_path="/home/nicolaedrabcinski/sd_kaggle/src"
    )
    
    generator = RealSubmissionGenerator()
    
    try:
        generator.load_real_data()  # Теперь загрузит реальный test.csv
    except FileNotFoundError as e:
        print(f"❌ Критическая ошибка: {e}")
        return []
    
    # ИСПОЛЬЗУЕМ РЕАЛЬНЫЕ ТЕСТОВЫЕ ДАННЫЕ
    test_data = generator.get_test_data()
    print(f"🎯 Используем тестовые данные: {test_data.shape}")
    print(f"📊 Первые 3 строки test данных:")
    print(test_data.head(3))
    
    successful_submissions = []
    
    for model_name in top_models:
        print(f"\n{'='*60}")
        print(f"🎯 ОБРАБОТКА: {model_name}")
        print(f"{'='*60}")
        
        # Загружаем модель
        model = loader.load_model(model_name)
        if model is None:
            continue
        
        # Генерируем submission НА РЕАЛЬНЫХ ТЕСТОВЫХ ДАННЫХ
        submission_df = generator.generate_submission_for_model(model, model_name, test_data)
        if submission_df is None:
            continue
        
        # Сохраняем
        output_path = generator.save_submission(submission_df, model_name)
        successful_submissions.append((model_name, output_path))
        
        print(f"✅ {model_name} - ЗАВЕРШЕНО")
    
    # Итоговый отчет
    print(f"\n{'='*60}")
    print("🎉 ИТОГОВЫЙ ОТЧЕТ")
    print(f"{'='*60}")
    print(f"✅ Успешно сгенерировано: {len(successful_submissions)}/{len(top_models)}")
    
    for model_name, path in successful_submissions:
        print(f"   📁 {model_name}: {path}")
    
    return successful_submissions

# Запуск генерации с реальными тестовыми данными
print("🔍 Проверяем наличие test.csv...")
test_path = Path("/home/nicolaedrabcinski/sd_kaggle/data/raw/test.csv")
if test_path.exists():
    print(f"✅ test.csv найден: {test_path}")
    submissions = generate_all_submissions_with_real_test()
    
    print(f"\n📋 ДАЛЬНЕЙШИЕ ШАГИ:")
    print("1. Проверьте сгенерированные файлы в папке /home/nicolaedrabcinski/sd_kaggle/submissions/")
    print("2. Для отправки на Kaggle используйте команды:")
    for model_name, path in submissions:
        print(f"   kaggle competitions submit -c mitsui-commodity-prediction-challenge -f {path} -m 'Submission with {model_name} - Dir Acc из ваших результатов'")
    print("3. Сравните результаты на лидерборде!")
else:
    print(f"❌ test.csv не найден по пути: {test_path}")
    
