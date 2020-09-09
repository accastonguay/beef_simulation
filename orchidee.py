folder = '/media/sf_OneDrive_-_The_University_of_Queensland/beef_project/datasets/ORCHIDEE/ORCHIDEE-GM/'
rasters = glob(folder + '*.nc')
# print(rasters)
for i in rasters:
    test = i.split("_GI_")[1].split(".n")[0]
#     print(i)
    print(test)
    ds = xr.open_dataset(i,decode_times=False)
    print(ds.coords)
    m = ds.BIOMASS_GRAZED.mean(axis = 0)/100.
    n = m.values
#     m.to_netcdf(path = '/home/adcas/Desktop/test.nc')


# #     ds.close()
    with rasterio.open('netcdf:'+i+':BIOMASS_GRAZED') as src:
        meta = src.meta
        print(meta)
        meta.update(driver='GTiff',
                   crs = rasterio.crs.CRS.from_epsg(3005))
        print(meta)

    with rasterio.open(folder + 'BIOMASS_GRAZED' + '_' + test + '.tif', 'w', **meta) as dst:
        dst.write(n, 1)