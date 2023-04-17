# ITUDeprem

- **efd_er_data.ipynb :** Views, processes and transforms EFD & ER data. You need to download and include station ".dat" files into _/data_ directory in project folder. [Station Data](https://drive.google.com/drive/folders/1pVy-RE3VjO5KtezpMcaVnX1QcUKLBcC5?usp=share_link)
- **er_mag_regulator.ipynb :** Manipulates  downloaded earthquake data (_\data\2002-01-01_2013-01-01_1.5_9.0(1).txt and (2)_) w.r.t _data\marmara_rect.geojson_. At the end, two output files energy_release_csv and magnitudes.csv are generated into data directory. The content criteria for both csv files are :
     - 1.5<xM, Depth<=20, between the years [2002-2013)
     - Locations inside Marmara Region ---> see exact area [AreaOfInterest](https://github.com/SrknMcT/ITUDeprem/blob/main/graphs/area-of-interest.jpg)
     - Energy=10^(5.24 + 1.44 * xM) in terms of megajoule
     - If more than one earthquake occurs in a day, only the one with highest magnitude is taken into account
- **foF2DataFormatter.ipynb :** Processes Ionospheric data and extracts new dataset to predict foF2 anomalies. You need to download and include an ionospheric data ".txt" file into _/data_ directory in project folder. Also make sure of having _Dst_2000_2023.txt_ in _/data_ as well. [Ionospheric Data](https://drive.google.com/drive/folders/1LKGH8AIc350Z0QEkC5Yr-pYMkvqsMnoW?usp=share_link) . <sub>( An article about that is under _/articles_ directory in project folder :  [Ionospheric Dısturbance]( https://github.com/SrknMcT/ITUDeprem/blob/main/articles/Ionospheric_foF2_disturbance_forecast.pdf) )</sub>

     <sub>**/data :** Includes some kind of data used in this project</sub><br><sub>**/graphs :** Includes some kind of 500 dpi plots of the codes</sub><br><sub>**/pecnet :** Includes data manipulation and error compensated neural network framework</sub><br><sub>**/articles :** Includes papers used in this project</sub>    

