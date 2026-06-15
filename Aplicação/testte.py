import cdsapi
meses = ["01","02","03","04","05","06","07","08","09","10","11","12"]
for ano in range(2015, 2020):
    for mess in meses:
        dataset = "reanalysis-era5-single-levels"

        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "2m_temperature",
                "2m_dewpoint_temperature",
                "surface_pressure",
                "total_precipitation",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "total_cloud_cover"
            ],
            "year": [str(ano)],
            "month": [
                mess
            ],
            "day": [
                "01","02","03","04","05","06","07",
                "08","09","10","11","12","13","14",
                "15","16","17","18","19","20","21",
                "22","23","24","25","26","27","28",
                "29","30","31"
            ],
            "time": [
                "00:00","01:00","02:00","03:00",
                "04:00","05:00","06:00","07:00",
                "08:00","09:00","10:00","11:00",
                "12:00","13:00","14:00","15:00",
                "16:00","17:00","18:00","19:00",
                "20:00","21:00","22:00","23:00"
            ],

            # Área EUA + Brasil
            # [Norte, Oeste, Sul, Leste]
            "area": [6, -75, -35, -30],

            "data_format": "netcdf",
            "download_format": "unarchived"
        }

        client = cdsapi.Client(url= "https://cds.climate.copernicus.eu/api", key= "2638aa06-1f91-4d0a-8ae2-d5cd72b7affc")

        client.retrieve(
            dataset,
            request,
            "era5BR_"+str(ano)+"_"+mess+".nc"
        )

        print("Download concluído.")