import train_parser as tp
import optimal_noise_k_minus_1 as optimal_n
import k_minus_1_noise_maker as noise_maker
import mst_wrapper as mst_wrapper
import logging
import os
LOG_PATH = '../../'


class exec_process():

    def __init__(self, create_files = False, train_models = True,log_file_name = LOG_PATH+'log_process'):
        self.create_files =  create_files
        self.log_file_name = log_file_name
        self.train_models = train_models
        if (os.path.isfile(log_file_name)):
            os.remove(log_file_name)

        logging.basicConfig(filename=self.log_file_name,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

    def execute(self):
        logging.info('Execution started')
        if (self.create_files):
            logging.info('Strat creating per language files')
            tp.create_files_per_language()
            logging.info('Finish creating per language files')

        if (self.train_models):
            logging.info('-'*50)
            logging.info('Training process started')
            tp.train_parser_all_lng()
            logging.info('Training process ended')


        logging.info('-' * 50)
        logging.info('Optimal noise learning started')
        optimal_n.find_optimal_noise_per_language()
        logging.info('Optimal noise learning ended')

        logging.info('-' * 50)
        logging.info('Noised dependency trees creation started')
        noise_maker.create_noised_dps_over_all_languages()
        logging.info('Noised dependency trees creation ended')

        logging.info('-' * 50)
        logging.info('mst wrapper started')
        mst_wrapper.mst_wrapper_for_all_languages()
        logging.info('Nmst wrapper ended')


exec_process_obj = exec_process(create_files=False,train_models= False)
exec_process_obj.execute()