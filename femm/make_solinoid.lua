function mi_line(start, stop)
    mi_addnode(start[1], start[2])
    mi_addnode(stop[1], stop[2])
    mi_addsegment(start[1], start[2], stop[1], stop[2])
end

function make_rect(inner_radius, height, thickness, y0, turns, material)
    local a = {inner_radius,             -height / 2 + y0}
    local b = {inner_radius + thickness, -height / 2 + y0}
    local c = {inner_radius + thickness,  height / 2 + y0}
    local d = {inner_radius,              height / 2 + y0}
    local center = {inner_radius + thickness/2, y0}
    mi_line(a, b)
    mi_line(b, c)
    mi_line(c, d)
    mi_line(d, a)
    mi_addblocklabel(center[1], center[2])
    return center
end

function make_solinoid(inner_radius, height, thickness, y0, turns, material)
    local center = make_rect(inner_radius, height, thickness, y0)
    mi_selectlabel(center[1], center[2])
    mi_setblockprop(material, 0, 0.01, "Coil", 0, 0, turns)
    return center 
end

function make_plunger(inner_radius, height, thickness, y0, material, magnitization_dir)
    local center = make_rect(inner_radius, height, thickness, y0)
    mi_selectlabel(center[1], center[2])
    mi_setblockprop(material, 0, 0.01, nil, 0, 0, 1)
    return center 
end

function mi_run_case(target_file, solinoid, core, current)
    open(target_file)
    mi_saveas("temp.fem")

    local solinoid = 
        make_solinoid(solinoid[1], solinoid[2], solinoid[3], 0.0, solinoid[4], "20 AWG")
    local plunger = 
        make_plunger(core[1], core[2], core[3], core[4], "N42", -90)
    
    
    mi_modifycircprop("Coil", 1, current) 

    mi_analyze()
    mi_loadsolution()

    local current, volts, flux_re = mo_getcircuitproperties("Coil")
    
    mo_selectblock(plunger[1], plunger[2])
    local force = mo_blockintegral(19)

    mo_close()
    mi_close()
    return force, current, volts, flux_re
end

function mi_compute_force_vs_z(target_file)
    local solinoid = {
        0.5, -- radius
        0.25, -- height
        0.125, -- thickness
        nil, -- turns
    }
    local awg_diam_mm = {
        [10] = 2.67716,
        [11] = 2.39268,
        [12] = 2.13868,
        [13] = 1.91516,
        [14] = 1.7145,
        [15] = 1.53162,
        [16] = 1.36652,
        [18] = 1.09474,
        [20] = 0.87884,
    }

    
    local diam = awg_diam_mm[20] * 0.0393701
    solinoid[4] = solinoid[2] * solinoid[3] / (diam * diam)
    print("param_num_turns=" .. solinoid[4])

    local gap = 0.01

    -- perc, p_y0
    --  0.0,    0
    --  1.0, -solinoid_height - plunger_height/2
    local norm = -solinoid_height - plunger_height/2

    -- do a first run to get the coil resistance
    local core = {
        0.125, -- radius
        0.25, -- height
        nil, -- thickness
        0, -- y0
    }
    core[3] = solinoid[1] - gap - core[1]

    local force, current, volts, flux_re = mi_run_case(target_file, solinoid, core, 1)
    local resistance = volts / current

    current = min(10, 7.4 / resistance)
    print("debug using colil current " .. current)




    for perc = -0.5, 0.5, 1/32 do
        core[4] = norm * perc
        
        local force, current, volts, flux_re = mi_run_case(target_file, solinoid, core, current)
        print(perc .. "," .. force)
        local resistance = volts / current
    end
end



clearconsole()
showconsole()
local target_file = mydir .. "bear_coaxial_solinoid.FEM"

mi_compute_force_vs_z(target_file)