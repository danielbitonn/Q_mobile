def __init__():
    IN_COLAB = False
    PATH = "--"
    ENCOD = "utf-8"
    try:
      import google.colab
      IN_COLAB = True
      PATH = r"/content/Q_mobile/"
    except:
      IN_COLAB = False
      PATH = ""

    print('In G-colab environment? ', IN_COLAB)
    R=5
    k=3
    print(R)
    
__init__()
################################################################################
################################################################################
################################################################################
import yaml
from yaml.loader import *

def func_get_conf(key_data):
    try:
        return yaml.load(open("utilities\_conf.yaml", 'r'), Loader=FullLoader)[key_data]
    except yaml.YAMLError as e:
        logging.error(f"_conf.yaml: {e}")
        raise Exception


################################################################################
################################################################################
################################################################################

def mf_quick_analysis(df, sweetviz=False):
  """
  DataTypes, Rows and Columns ,Null values, Unique values ...
  """
  print(" >>> Data info:")
  print(df.info())
  print("\n-------------****----------------\n\n >>> Null Values:")
  print(df.isnull().sum())
  print("\n-------------****----------------\n\n >>> Precentage of Nulls Values:")
  print(df.apply(lambda x: (sum(x.isnull()) / len(df)) * 100))
  print("\n-------------****----------------\n\n >>> Unique values:")
  print(df.nunique())
  print("\n-------------****----------------\n\n >>> Describes:")
  print(df.describe())
  print("\n-------------****----------------\n\n >>> Rows and Columns:")
  print(df.shape)
  print("\n-------------****----------------\n\n >>> The most often appears (.mode()) in the categorical columns, and the average (.mean()) for the continuous")
  for column in df.columns:
    if df[column].dtype == 'int64' or df[column].dtype == 'float64':
      avg = df[column].mean()
      print(f'{column} :  {avg}')
    elif df[column].dtype == 'O':
      mode = df[column].mode()[0] 
      print(f'{column} :  {mode}')
  if sweetviz:
    rep = sv.DataframeReport(df)
    rep.show_notebook()
    rep.show_html()
    return rep
  
################################################################################
################################################################################
################################################################################

def mf_get_files_from_git(_fileName, _fileLink):
  """
  *including extension
  with open({_fileName}, 'w') as f:
    f.write(requests.get({_fileLink}).text)
  """
  new_dir = 'utilities'
  try: 
    os.mkdir(f'{new_dir}')
  except FileExistsError: 
    pass

  os.chdir(f'{os.getcwd()}\\{new_dir}')
  
  with open(f'{_fileName}', 'w') as f:
    f.write(requests.get(f'{_fileLink}').text)
  print(f'\n{_fileName} has been created!')
  
  os.chdir(f'../')
################################################################################
################################################################################
################################################################################


class myc_classify_features:
  """
  This class support to identify the features types as a preperatoin for the EDA pahse.
  """
  

  def __init__(self):
                    """
                    numeric:      'dataTypes':[np.int64, np.float64]
                    continuous    # Floats
                    nominal       # Integers
                    
                    qualitative:  'dataTypes':[object, str, np.int64]
                    categorical   # Groups,
                    ordinal       # Rank, 
                    boolean       # True/False,
                    Binomial      # 0/1, True/False, Positive/Negative, heads/tails(coin)
                    discrete      # IDs
                    
                    timeSeries:   'dataTypes':[np.datetime64],
                    datetime      #
                    timedelta     #
                    objecttime    #

                    other:      'dataTypes':[ ]
                    other       #
                    garbage     #
                    target      #

                    """
                    
                    self.q_flag = 0
                    
                    self.feat_clf = {
                        'numeric': {    
                                        'continuous'  :set(),     # Floats
                                        'nominal'     :set()      # Integers
                        },
                        'qualitative': {
                                        'categorical'  :set(),    # Groups,
                                        'ordinal'     :set(),     # Rank, 
                                        'boolean'     :set(),     # True/False
                                        'Binomial'    :set(),     # 0/1, Positive/Negative, heads/tails(coin)
                                        'discrete'    :set()      # IDs,
                        },
                        'timeSeries': { 
                                        'datetime'    :set(),    
                                        'timedelta'   :set(), 
                                        'objecttime'  :set()
                        },
                        'other':      {
                                        'other'       :set(),
                                        'garbage'     :set(),
                                        'target'      :set()
                        }
                    }
                    
                    self.feat_clas = [clas for clas, item in self.feat_clf.items()]
                    self.feat_sub_clas = [[clas,typ] for clas in self.feat_clas for typ, cont in self.feat_clf[clas].items()]
                    # pprint 
                    # self.mycf_pprint()


  def mycf_pprint(self):
    for clas, item in self.feat_clf.items():
        print(clas,':')  
        for typ, content in item.items():
          if len(content)==0:
            content=''
            print(typ, ' >>> ' ,content)
          else:
            print(typ, ' >>> ' ,content)
        print()


  def mycf_update_feat_clf(self, col_name):
    """
    class: 'numeric', 'qualitative', 'timeSeries', 'other' 
    """
    # Init
    self.col_name = col_name;
    # Functions
    self._user_input()
    self._append_column()

    
  def _user_input(self):
    print(f'Choose the dataType of the column\n{self.col_name}\
            \nChoose a Number from the following:\n')
    for i, x in enumerate(self.feat_sub_clas):
      print(i, x)
    try:
      self.input = input('\n\n >>> Press Q/q to stop.\n')
    except KeyboardInterrupt:
      self.input = self.feat_sub_clas.index(["other", "other"])

    if self.input=='Q' or self.input=='q':
      self.q_flag=1;
    elif self.input=='' or int(self.input) >= len(self.feat_sub_clas):
      self._user_input()

  def _append_column(self):
    """
    #No >>> ['class', 'sub_class']
    """
    if self.q_flag==1:
      pass
    else:
      self.feat_clf[self.feat_sub_clas[int(self.input)][0]][self.feat_sub_clas[int(self.input)][1]].add(self.col_name)


################################################################################
################################################################################
################################################################################


class myc_numeric_eda:
  """
  This class organize the methods in EDA phase. 
  """
  def __init__(self):
    """
    Statistics: [mean, std, quartiles, distributions]
    Methods:    [counting, frequencies, binarization, rounding, Fixed-Width Binning, Adaptive Binning, log Transform, Box-Cox Transform]
    """


  def mycf_Binarization(self, df, counted_col, f_ts: float=0.9, i_ts: int=1, meth: int=1):
    """
    Conted column is required !!!
    -----------------------------
    meth:
    1     | manually
    2     | sklearn.Binarizer
    thresholds: 
    f_ts  | 0.1 to 1.0
    i_ts  | 1   to 999
    """
    self.f_ts = f_ts
    self.i_ts = i_ts
    df = df.copy()

    if meth==1:
      npa_col = np.array(df[counted_col]) 
      npa_col[npa_col >= self.i_ts] = 1
      df[counted_col] = npa_col

    elif meth==2:
      from sklearn.preprocessing import Binarizer
      bn = Binarizer(threshold=self.f_ts)
      df[counted_col] = bn.transform([df[counted_col]])[0]

    return df.reset_index(drop=True)
  

