import pandas as pd

class Environment:
    def __init__(self):   
        self.IN_COLAB = False
        self.PATH = "--"
        self.ENCOD = "utf-8"
        try:
          import google.colab
          self.IN_COLAB = True
          self.PATH = r"/content/Q_mobile/"
        except:
          self.IN_COLAB = False
          self.PATH = ""

        self.rawdata = pd.read_csv(f'data/data_for_candidate.csv', parse_dates=['TIME'])
        
# ENV = Environment()

class ExplorDataAnaly:
    def __init__(self):
        pass
    def date_time_handling(self, df, date_time_col , 
                                       interval='1H', is_weekend=4,
                                       morning_start=6, morning_end=12,
                                       noon_start=12, noon_end=18, 
                                       evening_start=18, evening_end=22):
        
        """
        1. This function takes a data frame df and a column name date_time_col representing the date and time column. 
        2. It also takes the start and end hours for: [morning, noon, evening, and night] 
           default values: [6 for morning start, 12 for morning end, 12 for noon start, 18 for noon end, 18 for evening start, and 22 for evening end]
           creates columns for morning, noon, evening, and night based on the hour of the day.
        3. The function first cleans missing and illegal values by converting the column to a datetime type and removing any rows with missing values. 
        4. Then extracts the [year, month, day, hour, weekday, and weekday] 
           creates a weekday and weekend columns.  
        5. column which represent the relative time due to the first timestamp of the data as point 0 then in flexible interval 
           (for example: xx_H) continuously relative to time 0 intervals are:
           ['xx_H'/'xx_M'/'xx_D'/'xx_W'/'xx_M' end so forth].
        6. Insert timeStamp column
        """
        self.df             = df.copy()
        self.interval       = interval
        self.is_weekend     = is_weekend
        self.date_time_col  = date_time_col
        self.morning_start  = morning_start
        self.morning_end    = morning_end
        self.noon_start     = noon_start
        self.noon_end       = noon_end
        self.evening_start  = evening_start
        self.evening_end    = evening_end
        """
        Note! 
        You should define a variable as a self.var when it's an instance variable. 
        Instance variables are variables that belong to an instance of a class, 
        and they have different values for different instances of the same class.
        In other words, if you have a class MyClass, and you create two instances of it, obj1 and obj2, 
        each of these instances will have its own copy of the instance variables defined in the class. 
        If you modify the value of an instance variable for one instance, 
        it will not affect the value of the same variable for the other instance.
        """

        # Clean missing and illegal values
        df[self.date_time_col] = pd.to_datetime(df[self.date_time_col], errors='coerce')
        df = df.dropna(subset=[self.date_time_col])

        # Insert timeStamp
        df['timeStamp'] =  df[self.date_time_col].apply(lambda x: x.timestamp())

        # Get the first timestamp as reference point
        self.first_timestamp = df[self.date_time_col].min()
        df['relative_time'] = (df[self.date_time_col] - self.first_timestamp).dt.total_seconds()

        # Split the interval string into quantity and abbreviation
        quantity = int(interval[:-1])
        abbreviation = interval[-1]
        if abbreviation   == 'H':
            interval_seconds = quantity * 3600
        elif abbreviation == 'M':
            interval_seconds = quantity * 60
        elif abbreviation == 'D':
            interval_seconds = quantity * 3600 * 24
        elif abbreviation == 'W':
            interval_seconds = quantity * 3600 * 24 * 7
        else:
            raise ValueError("Invalid interval abbreviation")

        # Convert the relative time to an integer representing the interval
        df['relative_time'] = df['relative_time'] // interval_seconds
        df['relative_time'] = df['relative_time'].astype(int)

        # Extract year, month, day, hour, and weekday
        df['year']    = df[self.date_time_col].dt.year
        df['month']   = df[self.date_time_col].dt.month
        df['dateDay'] = df[self.date_time_col].dt.day
        df['weekDay'] = df[self.date_time_col].dt.weekday
        df['nameDay'] = df[self.date_time_col].dt.day_name()
        df['hour']    = df[self.date_time_col].dt.hour

        # Create a weekend column
        df['weekend'] = np.where((df['weekDay'] >= is_weekend) & (df['weekDay'] < is_weekend+2), 1, 0)    
        ### TODO: insert transforme for change the weekend (israel) including the nameDay.

        # Create columns for morning, noon, evening, and night
        df['morning'] = ((df['hour'] >= morning_start) & (df['hour'] < morning_end)).astype(int)
        df['noon']    = ((df['hour'] >= noon_start) & (df['hour'] < noon_end)).astype(int)
        df['evening'] = ((df['hour'] >= evening_start) & (df['hour'] < evening_end)).astype(int)
        df['night']   = ((df['hour'] < morning_start) | (df['hour'] >= evening_end)).astype(int)

        # Drop the original date time column
        df = df.drop(columns=[self.date_time_col])

        return df
        # return df[['timeStamp' ,'relative_time','year', 'month', 'dateDay', 'nameDay', 'weekDay', 'weekend', 'hour', 'morning', 'noon', 'evening', 'night']]

    def distance_haversine(self, lat1, lon1, lat2, lon2):
      # Convert latitude and longitude to radians
      lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
      # Calculate the difference in latitude and longitude
      dlat = lat2 - lat1 
      dlon = lon2 - lon1
      # Use the Haversine formula to calculate the distance
      a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
      c = 2 * math.asin(math.sqrt(a))
      distance = 6371 * c # 6371 is the radius of the Earth in kilometers
      return distance #*1000 # convert to m

    
    def func_mapping_by_categorical_feat(df,  cat_feat, sub_cat_feat='', rad='', coord_list=['GEO_LAT', 'GEO_LON']):
      if sub_cat_feat != '':
        df = df[df[cat_feat]==sub_cat_feat].copy()
      else:
        df = df.copy()

      if rad != '':
        rad_flag=True
      else:
        rad_flag=False

      colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 
              'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', 
              'black', 'lightgray', 'red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 
              'lightgreen']

      # Create a Folium map centered on the mean of the longitudes and latitudes
      mean_lat = df['GEO_LAT'].mean()
      min_lat = df['GEO_LAT'].min()
      max_lat = df['GEO_LAT'].max()

      mean_lon = df['GEO_LON'].mean()
      min_lon = df['GEO_LON'].min()
      max_lon = df['GEO_LON'].max()
      x = 7

      # location = df[coord_list].values.tolist()
      feat_values = df[cat_feat].tolist()

      # Create a dictionary that maps countries to colors
      feat_colors = defaultdict(lambda: 'blue')
      unique_feat_values = set(feat_values)
      for i, feat in enumerate(unique_feat_values):
          # feat_colors[feat] = ['red', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen', 'cadetblue','lightred', 'black', 'darkpurple', 'pink'][i % 13]
          feat_colors[feat] = colors[i % 13]

      m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12, tiles='OpenStreetMap', 
                   zoom_control=False, disable_3d=True,  no_touch=True,
                   min_lat=min_lat-x, max_lat=max_lat+x, min_lon=min_lon-x, max_lon=max_lon+x
                   )
      # m = folium.Map(location=location[0], zoom_start=10)


      if rad_flag:
        for lat, lon, feat, rad in zip(df[coord_list[0]], df[coord_list[1]], df[cat_feat], df[rad]):
            folium.CircleMarker(
                                location=[lat, lon],
                                radius=rad,
                                color=feat_colors[feat],
                                fill=True,
                                fill_color=feat_colors[feat]
                            ).add_to(m)
      else:
        for lat, lon, feat in zip(df[coord_list[0]], df[coord_list[1]], df[cat_feat]):
            folium.CircleMarker(
                                location=[lat, lon],
                                radius=1,
                                color=feat_colors[feat],
                                fill=True,
                                fill_color=feat_colors[feat]
                            ).add_to(m)

      # Add a legend to the map for only the countries that have points
      legend_html = '<div style="position: fixed; bottom: 50px; left: 50px; width: 150px; height: auto; border:2px solid grey; z-index:9999; font-size:14px; padding: 10px;">'
      legend_html += f'<b>{cat_feat} Legend</b><br>'
      for feat in unique_feat_values:
          color = feat_colors[feat]
          legend_html += f'&nbsp; {feat} &nbsp; <i class="fa fa-circle fa-xs" style="color:{color}"></i><br>'
      legend_html += '</div>'

      m.get_root().html.add_child(folium.Element(legend_html))
      return m

    
# EDA = ExplorDataAnaly()

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
  

