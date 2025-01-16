import csv


def _logger(log, stage, csv_filename = './'):
    lines = log.split("\n")
    data = []
    
    for line in lines:
        if line.startswith("Epoch"):
            epoch = int(line.split()[1].split('/')[0]) - 1
        elif line.startswith(stage):
            loss = float(line.split("|")[0].split(":")[1].strip())
            data.append([1735105736, epoch, loss])

    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Wall time", "Step", "Value"])  
        writer.writerows(data)

    print(f"Данные успешно сохранены в файл {csv_filename}.")


# _logger(log, 
#         "TRAIN", 
#         csv_filename = "./checkpoints/dualpathrnn/train_loss.csv")

