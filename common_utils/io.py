import csv
import pickle
import logging


def load_pickle(filename):
    try:
        with open(str(filename), 'rb') as f:
            obj = pickle.load(f)

        logging.info('Loaded: %s', filename)

    except EOFError:
        logging.warning('Cannot load: %s', filename)
        obj = None

    return obj


def save_pickle(filename, obj):
    with open(str(filename), 'wb') as f:
        pickle.dump(obj, f)

    logging.info('Saved: %s', filename)


def save_csv(filename, data, fieldnames=None, flush=False):
    with open(str(filename), 'w') as f:
        if fieldnames is not None:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        else:
            writer = csv.writer(f)

        if not flush:
            writer.writerows(data)
        else:
            # write line by line and flush after each line
            for row in data:
                writer.writerow(row)
                f.flush()

    logging.info('Saved: %s', filename)


def load_csv(filename, header=True):
    with open(str(filename), 'r') as f:
        if header:
            reader = csv.DictReader(f)
        else:
            reader = csv.reader(f)

        rows = [r for r in reader]

    logging.info('Loaded: %s', filename)
    return rows


def load_lines(filename):
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f]
        lines = [l for l in lines if len(l) != 0]

    logging.info('Loaded: %s', filename)
    return lines
