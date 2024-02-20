from utils.bag_utils import BagReview
path = 'tomas\\20230511_102553.bag'
output_path = 'output'
bag = BagReview(path)
bag.save_all_frames(output_path)