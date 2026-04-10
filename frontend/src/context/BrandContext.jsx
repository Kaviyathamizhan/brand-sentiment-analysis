import React, { createContext, useState, useRef } from 'react';

export const BrandContext = createContext();

export const BrandProvider = ({ children }) => {
    // We use a global ref holding our network dictionary cache.
    // This allows components to instantly resolve data without re-fetching upon swapping tabs.
    const metricsCache = useRef({});

    const [globalAlerts, setGlobalAlerts] = useState([]);
    
    return (
        <BrandContext.Provider value={{
            metricsCache,
            globalAlerts, setGlobalAlerts
        }}>
            {children}
        </BrandContext.Provider>
    );
};
