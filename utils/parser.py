import csv


def _logger(logs, stage, index=0, csv_filename = './'):
    lines = logs.split("\n")
    data = []
    
    for line in lines:
        if line.startswith("Epoch"):
            epoch = int(line.split()[1].split('/')[0]) - 1
        elif line.startswith(stage):
            val = float(line.split("|")[index].split(":")[1].strip())
            data.append([1735105736, epoch, val])
            
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Wall time", "Step", "Value"])  
        writer.writerows(data)

    print(f"Данные успешно сохранены в файл {csv_filename}")


# _logger(log, 
#         "TRAIN", 
#         csv_filename = "./checkpoints/dualpathrnn/train_loss.csv")

