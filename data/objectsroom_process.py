import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf

import third_party.multi_object_datasets.objects_room as objects_room


data_root = '/scratch/generalvision/ObjectsRoom/'
OBJECTS_ROOM = 'objects_room_train.tfrecords'
SEED = 0
num_workers = 4
num_entities = 7
background_entities = 4
total_sz = 1000000
val_sz = 10000
tst_sz = 10000
batch_size = 2000

# save img and mask as png
def save_img_and_mask(dataset, batch_size, num_img, background_entities, num_entities, split):
    if not os.path.exists(f'{data_root}/{split}/images'):
        os.makedirs(f'{data_root}/{split}/images')
    if not os.path.exists(f'{data_root}/{split}/masks'):
        os.makedirs(f'{data_root}/{split}/masks')
    batched_dataset = dataset.batch(batch_size)  # optional batching
    iterator = batched_dataset.make_one_shot_iterator()
    data = iterator.get_next()
    N = num_img // batch_size
    with tf.compat.v1.Session() as sess:
        num = 0
        for _ in tqdm(range(N)):
            batch = sess.run(data)
            image_batch = batch.pop('image')
            raw_masks = batch.pop('mask')
            B, H, W, C = image_batch.shape
            # Parse masks
            masks = np.zeros((B, 1, H, W), dtype='int')
            # Convert to boolean masks
            cond = np.where(raw_masks[:, :, :, :, 0] == 255, True, False)
            # Ignore background entities
            num_entities = cond.shape[1]
            for o_idx in range(background_entities, num_entities):
                masks[cond[:, o_idx:o_idx+1, :, :]] = o_idx + 1
            for i in range(B):
                # save image
                img = image_batch[i]
                img = Image.fromarray(img, 'RGB')
                image_path = f'{data_root}/{split}/images/img_{str(num).zfill(6)}.png'
                img.save(image_path)
                # save mask
                mask = masks[i].reshape(H, W) # (64, 64, 1)
                # save other properties
                mask = Image.fromarray(mask.astype(np.uint8), 'P')
                palette = [random.randint(0, 255) for x in range(256 * 3)]
                mask.putpalette(palette)
                mask_path = f'{data_root}/{split}/masks/mask_{str(num).zfill(6)}.png'
                mask.save(mask_path)
                num += 1
            if B < batch_size:
                print(f'processed {num}  {split} images')
                break
     
# load data from tfrecords and save as png
def process_data():

    # Fix TensorFlow seed
    global SEED
    tf.random.set_seed(SEED)
    tf.compat.v1.disable_eager_execution()

    data_path = data_root + OBJECTS_ROOM
    raw_dataset = objects_room.dataset(
        data_path,
        'train',
        map_parallel_calls=num_workers)
    # Split into train / val / test
    raw_dataset = raw_dataset.take(total_sz)
    print(f"Dataset has {total_sz} frames")
    
    tng_sz = total_sz - val_sz - tst_sz
    print(f"Splitting into {tng_sz}/{val_sz}/{tst_sz} for tng/val/tst")
    tst_dataset = raw_dataset.take(tst_sz)
    val_dataset = raw_dataset.skip(tst_sz).take(val_sz)
    tng_dataset = raw_dataset.skip(tst_sz + val_sz)

    # save img and mask
    print('saving img and mask')
    save_img_and_mask(tng_dataset, batch_size, tng_sz, background_entities, num_entities, 'train')
    save_img_and_mask(val_dataset, batch_size, val_sz, background_entities, num_entities, 'val')
    save_img_and_mask(tst_dataset, batch_size, tst_sz, background_entities, num_entities, 'test')
    print("Done")


'''load data from tfrecords and save as png, you only need to run it once'''
if __name__ == '__main__':
    process_data()

    
