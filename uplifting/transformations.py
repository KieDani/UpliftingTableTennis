import numpy as np
from uplifting.helper import HEIGHT, WIDTH
from uplifting.helper import cam2img, world2cam
from uplifting.helper import KEYPOINT_VISIBLE, KEYPOINT_INVISIBLE


class Compose:
    '''
    Composes several transforms together.
    '''
    def __init__(self, transforms):
        '''
        transforms (list of objects) – list of transforms to compose.
        '''
        self.transforms = transforms

    def __call__(self, data):
        '''
        data (dict) – Dictionary with keys 'r_img' and 'table_img'.
        '''
        for t in self.transforms:
            data = t(data)
        return data


class RandomizeDetections:
    '''
    Randomize image coordinates to simulate noisy detections.
    '''
    def __init__(self, std=5):
        '''
        seed (int) – Seed for the random number generator.
        '''
        self.std = std

    def __call__(self, data):
        '''
        data (dict) – Dictionary with keys 'r_img' and 'table_img'.
        '''
        r_img = data['r_img']
        table_img = data['table_img']
        r_img = r_img + np.random.normal(loc=0, scale=self.std, size=r_img.shape)
        table_img[:, :2] = table_img[:, :2] + np.random.normal(loc=0, scale=self.std, size=table_img[:, :2].shape)
        data['r_img'] = r_img
        data['table_img'] = table_img
        return data


class RandomStop:
    '''
    Randomly stop the sequence after the bounce to simulate that the oposing player hit the ball.
    '''
    def __init__(self, stop_prob=0.5):
        '''
        stop_prob (float) – Probability to stop the sequence after the bounce.
        '''
        self.stop_prob = stop_prob

    def __call__(self, data):
        '''
        data (dict) – Dictionary with keys 'r_img' and 'table_img'.
        '''
        if np.random.random() > self.stop_prob: # don't do anything
            return data
        else:
            hits = data['hits']
            times = data['times']
            mask = data['mask']
            r_img = data['r_img']
            r_world = data['r_world']

            hit_time = hits[0]  # take the first bounce
            if hit_time > 0: # there is a bounce
                hit_ind = np.argmin(np.abs(times - hit_time))
                seq_len = np.sum(mask)
                if seq_len - hit_ind >= 4: # minimum sequence length after hit
                    len_after_hit = np.random.randint(4, seq_len - hit_ind + 1)
                    mask[hit_ind + len_after_hit:] = False
                    data['mask'] = mask
                    # set coordinates to 0 after the new end
                    r_img[mask == False] = r_img[mask == False] * 0
                    r_world[mask == False] = r_world[mask == False] * 0
                    times[mask == False] = times[mask == False] * 0
                    data['r_img'] = r_img
                    data['r_world'] = r_world
                    data['times'] = times

            return data


class MotionBlur:
    '''
    Simulate noisy detections due to motion blur by adding noise along the trajectory in image space. Should be applied before RandomizeDetections.
    '''
    def __init__(self, blur_strength=0.5):
        # if 1, the motion blur will be chosen from full range between the previous and next frame
        # if 0, there is basically no motion blur
        self.blur_strength = blur_strength
        assert 0.1 <= blur_strength < 0.5 or blur_strength == 0, 'blur_strength should be in the range [0.1, 0.5) or 0.'

    def __call__(self, data):
        '''
        data (dict) – Dictionary with keys 'r_img' and 'table_img'.
        '''
        if self.blur_strength == 0:
            return data
        r_worlds = data['r_world']  # Shape (T, 3)
        r_imgs = data['r_img']  # Shape (T, 2)
        Mint = data['Mint']
        Mext = data['Mext']
        mask = data['mask']
        length = np.sum(mask)
        times = data['times']  # Shape (T,)
        blur_r_world = data['blur_positions']  # Shape (T, 3)
        blur_times = data['blur_times']  # Shape (T,)
        # easily access the time of the previous and next frame
        before, after = times.copy(), times.copy()
        before[1:length] = times[:length-1]
        after[:length-1] = times[1:length]
        # evaluate the time boundary before and after (with the strength parameter)
        before[:length] = times[:length] + self.blur_strength * (before - times)[:length]
        after[:length] = times[:length] + self.blur_strength * (after - times)[:length]
        # iterate over trajectory (TODO: this is slow, can be optimized)
        for i in range(length):
            b, a = before[i], after[i]  # Scalars
            # get all the coordinates that are in the range of the blur
            valid_blur_times = blur_times[(blur_times >= b) & (blur_times <= a)]
            valid_blur_r = blur_r_world[(blur_times >= b) & (blur_times <= a)]
            # choose a random coordinate that is inside the blur range
            blur_t = np.random.choice(valid_blur_times)
            blur_r = valid_blur_r[valid_blur_times == blur_t]
            blur_r = blur_r[0]
            blur_r_cam = world2cam(blur_r, Mext)
            blur_r_img = cam2img(blur_r_cam, Mint)
            # update the r and r_img
            r_worlds[i] = blur_r
            r_imgs[i] = blur_r_img
        data['r_world'] = r_worlds
        data['r_img'] = r_imgs
        return data


class RandomDetection:
    '''
    Instead of the ball or table keypoint, a random image point is chosen.
    '''
    def __init__(self, randdet_prob):
        '''
        miss_prob (float) – Probability of a miss detection. Should be in the range [0, 1].
        '''
        self.randdet_prob = randdet_prob

    def __call__(self, data):
        '''
        data (dict) – Dictionary with keys 'r_img' and 'table_img'.
        '''
        r_img = data['r_img']
        table_img = data['table_img']
        mask = data['mask']

        T = np.sum(mask)
        for r_img_ind in range(T):
            if np.random.random() < self.randdet_prob:
                # choose a random point in the image
                r_img[r_img_ind] = np.random.rand(*r_img[r_img_ind].shape) * np.array([WIDTH, HEIGHT])

        for table_img_ind in range(table_img.shape[0]):
            if np.random.random() < self.randdet_prob:
                # choose a random point in the image
                table_img[table_img_ind, :2] = np.random.rand(2) * np.array([WIDTH, HEIGHT])
        data['r_img'] = r_img
        data['table_img'] = table_img
        return data


class RandomMissing:
    '''The ball was not detected in the image, so we remove the ball coordinates.'''
    def __init__(self, randmiss_prob):
        '''
        miss_prob (float) – Probability of a miss detection. Should be in the range [0, 1].
        '''
        self.randmiss_prob = randmiss_prob

    def __call__(self, data):
        '''
        data (dict) – Dictionary with keys 'r_img' and 'table_img'.
        '''
        r_img = data['r_img']
        r_world = data['r_world']
        times = data['times']
        mask = data['mask']

        new_mask = mask.copy()
        new_mask[:] = False
        new_r_img = np.zeros_like(r_img)
        new_r_world = np.zeros_like(r_world)
        new_times = np.zeros_like(times)

        T = np.sum(mask)
        current_index = 0  # index without removed frames
        for i in range(T):
            # else just leave it out
            if not (np.random.random() < self.randmiss_prob):
                new_mask[current_index] = True
                new_r_img[current_index] = r_img[i]
                new_r_world[current_index] = r_world[i]
                new_times[current_index] = times[i]
                current_index += 1

        # update the mask, r_img, r_world, and times
        data['mask'] = new_mask
        data['r_img'] = new_r_img
        data['r_world'] = new_r_world
        data['times'] = new_times
        return data


class TableMissing:
    '''Sometimes a table keypoint is not detected. Thus, we randomly set the visibility to 0.'''
    def __init__(self, tablemiss_prob):
        '''
        miss_prob (float) – Probability of a miss detection. Should be in the range [0, 1].
        '''
        self.tablemiss_prob = tablemiss_prob

    def __call__(self, data):
        '''
        data (dict) – Dictionary with keys 'r_img' and 'table_img'.
        '''
        table_img = data['table_img']
        for table_img_ind in range(table_img.shape[0]):
            if np.random.random() < self.tablemiss_prob:
                table_img[table_img_ind, 2] = KEYPOINT_INVISIBLE
                # Set the coordinates to random values within the image
                table_img[table_img_ind, :2] = np.random.rand(2) * np.array([WIDTH, HEIGHT])
        data['table_img'] = table_img
        return data



class Identity:
    '''
    Identity transform.
    '''
    def __init__(self):
        pass

    def __call__(self, data):
        return data


class NormalizeImgCoords:
    '''
    Normalize image coordinates to the range [0, 1] using HEIGHT and WIDTH. Apply transformation after RandomizeDetections!!!
    '''
    def __init__(self):
        pass

    def __call__(self, data):
        r_img = data['r_img']
        table_img = data['table_img']
        r_img = r_img / np.array([WIDTH, HEIGHT])
        table_img[..., :2] = table_img[..., :2] / np.array([WIDTH, HEIGHT])
        data['r_img'] = r_img
        data['table_img'] = table_img
        return data


class UnNormalizeImgCoords:
    '''
    Unnormalize image coordinates to the range [0, WIDTH] and [0, HEIGHT]. Apply transformation after RandomizeDetections!!!
    '''
    def __init__(self):
        pass

    def __call__(self, data):
        r_img = data['r_img']
        table_img = data['table_img']
        r_img = r_img * np.array([WIDTH, HEIGHT])
        table_img[..., :2] = table_img[..., :2] * np.array([WIDTH, HEIGHT])
        data['r_img'] = r_img
        data['table_img'] = table_img
        return data


def get_transforms(config, mode='train'):
    '''
    Get the transforms for the dataset.
    mode (str) – Mode of the dataset. Can be 'train', 'val', or 'test'.
    '''
    transforms = []
    if mode == 'train':
        transforms.append(MotionBlur(config.blur_strength))
        transforms.append(RandomizeDetections(config.randomize_std))
        transforms.append(RandomStop(config.stop_prob))
        transforms.append(RandomDetection(config.randdet_prob))
        transforms.append(RandomMissing(config.randmiss_prob))
        transforms.append(TableMissing(config.tablemiss_prob))
    transforms.append(NormalizeImgCoords())
    return Compose(transforms)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from uplifting.data import TableTennisDataset

    dataset = TableTennisDataset('val', None)
    data = dataset[1]
    r_img, table_img, mask, r_world, rotation, times, hits, Mint, Mext = data
    # plot r_img
    plt.figure()
    plt.plot(r_img[:10, 0], r_img[:10, 1], 'o')
    plt.xlim([r_img[:10, 0].min()-5, r_img[:10, 0].max()+5])
    plt.ylim([r_img[:10, 1].min()-5, r_img[:10, 1].max()+5])
    plt.title('Original')
    plt.show()

    transforms = MotionBlur(blur_strength=0.4)
    dataset_trans = TableTennisDataset('val', transforms)
    data = dataset_trans[1]
    r_img, table_img, mask, r_world, rotation, times, hits, Mint, Mext = data
    # plot r_img
    plt.figure()
    plt.plot(r_img[:10, 0], r_img[:10, 1], 'o')
    plt.xlim([r_img[:10, 0].min()-5, r_img[:10, 0].max()+5])
    plt.ylim([r_img[:10, 1].min()-5, r_img[:10, 1].max()+5])
    plt.title('MotionBlur')
    plt.show()

    transforms = RandomizeDetections(std=3)
    dataset_trans2 = TableTennisDataset('val', transforms)
    data = dataset_trans2[1]
    r_img, table_img, mask, r_world, rotation, times, hits, Mint, Mext = data
    # plot r_img
    plt.figure()
    plt.plot(r_img[:10, 0], r_img[:10, 1], 'o')
    plt.xlim([r_img[:10, 0].min()-5, r_img[:10, 0].max()+5])
    plt.ylim([r_img[:10, 1].min()-5, r_img[:10, 1].max()+5])
    plt.title('RandomizeDetections')
    plt.show()

    dataset = TableTennisDataset('val', None)
    data = dataset[2]
    r_img, table_img, mask, r_world, rotation, times, hits, Mint, Mext = data
    # plot r_img
    plt.figure()
    plt.plot(r_img[:, 0], r_img[:, 1], 'o')
    plt.xlim([r_img[:, 0].min() - 5, r_img[:, 0].max() + 5])
    plt.ylim([r_img[:, 1].min() - 5, r_img[:, 1].max() + 5])
    plt.title('Original')
    plt.show()

    transforms = RandomStop(stop_prob=1)
    dataset_trans3 = TableTennisDataset('val', transforms)
    data = dataset_trans3[2]
    r_img, table_img, mask, r_world, rotation, times, hits, Mint, Mext = data
    # plot r_img
    plt.figure()
    plt.plot(r_img[:, 0], r_img[:, 1], 'o')
    plt.xlim([r_img[:, 0].min() - 5, r_img[:, 0].max() + 5])
    plt.ylim([r_img[:, 1].min() - 5, r_img[:, 1].max() + 5])
    plt.title('RandomStop')
    plt.show()


