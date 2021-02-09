import torch
import time


class Logger(object):
    
    def __init__(self, runs, info=None, file_name=None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.file_name = file_name

    def add_result(self, run, result):
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
            return result[:, 1].max()
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                p_10 = r[r[:, 1].argmax(), 3].item()
                p_50 = r[r[:, 1].argmax(), 4].item()
                p_100 = r[r[:, 1].argmax(), 5].item()
                auc_score = r[r[:, 1].argmax(), 6].item()
                aupr_score = r[r[:, 1].argmax(), 7].item()
                f1_score = r[r[:, 1].argmax(), 8].item()
                best_results.append((train1, valid, train2, test, p_10, p_50, p_100, auc_score, aupr_score, f1_score))

            best_result = torch.tensor(best_results)

            if self.file_name is not None:
                file = open(self.file_name, 'a')
            else:
                file = None
            print(f'All runs:', file=file)
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}', file=file)
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}', file=file)
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}', file=file)
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}', file=file)
            r = best_result[:, 4]
            print(f'   Final Precision @ 10: {r.mean():.2f} ± {r.std():.2f}', file=file)
            r = best_result[:, 5]
            print(f'   Final Precision @ 50: {r.mean():.2f} ± {r.std():.2f}', file=file)
            r = best_result[:, 6]
            print(f'   Final Precision @ 100: {r.mean():.2f} ± {r.std():.2f}', file=file)
            r = best_result[:, 7]
            print(f'   Final AUC score: {r.mean():.2f} ± {r.std():.2f}', file=file)
            r = best_result[:, 8]
            print(f'   Final AUPR score: {r.mean():.2f} ± {r.std():.2f}', file=file)
            r = best_result[:, 9]
            print(f'   Final F1 score: {r.mean():.2f} ± {r.std():.2f}', file=file)
            if file is not None:
                file.close()
            return [ (best_result[:, 1].mean(), best_result[:, 1].std()), (best_result[:, 3].mean(), best_result[:, 3].std()) ]
            
