import torch
from torch.export import export
from torch_xla.stablehlo import exported_program_to_stablehlo

# 1. Импортируем или описываем архитектуру вашей модели
from my_model_file import MyNeuralNetwork

model = MyNeuralNetwork()

# 2. Загружаем веса из вашего .pth файла
model.load_state_dict(torch.load("my_model_weights.pth"))
model.eval()

# 3. Создаем пример входных данных (dummy input) нужной формы
sample_input = (torch.randn(1, 3, 224, 224),)

# 4. Экспортируем граф через стандартный torch.export
exported_program = export(model, sample_input)

# 5. Конвертируем в StableHLO и сохраняем
stablehlo_program = exported_program_to_stablehlo(exported_program)
stablehlo_program.save("/tmp/my_model_stablehlo")
