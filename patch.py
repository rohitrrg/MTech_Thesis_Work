with open('FourCastNet/generate_baseline.py', 'r') as f:
    code = f.read()

old_clim = """        print("  Computing Climatology...")
        clim_time = pd.to_datetime(clim_dataset.time.values)
        clim_time_2018 = pd.to_datetime(
            [f"2018-{t.strftime('%m-%d %H:%M:%S')}" for t in clim_time]
        )
        clim_aligned = clim_dataset.assign_coords(time=clim_time_2018)
        clim_step = clim_aligned.sel(time=valid_times, method="nearest")"""

new_clim = """        print("  Computing Climatology...")
        # Match time via dayofyear and hour matching WeatherBench2 climatology specs
        target_times = pd.to_datetime(valid_times.values)
        
        # Some climatology stores use time, some use dayofyear/hour
        if "dayofyear" in clim_dataset.coords:
            dayofyear = xr.DataArray(target_times.dayofyear, dims=["time"], coords={"time": valid_times})
            hour = xr.DataArray(target_times.hour, dims=["time"], coords={"time": valid_times})
            clim_sel = clim_dataset.sel(dayofyear=dayofyear, hour=hour)
            if "dayofyear" in clim_sel.dims:
                clim_step = clim_sel.swap_dims({"dayofyear": "time"})
            else:
                clim_step = clim_sel
        else:
            clim_time = pd.to_datetime(clim_dataset.time.values)
            clim_time_2018 = pd.to_datetime([f"2018-{t.strftime('%m-%d %H:%M:%S')}" for t in clim_time])
            clim_aligned = clim_dataset.assign_coords(time=clim_time_2018)
            clim_step = clim_aligned.sel(time=valid_times, method="nearest")"""

code = code.replace(old_clim, new_clim)

with open('FourCastNet/generate_baseline.py', 'w') as f:
    f.write(code)

print("Patched clim_step extraction")
