// i18n utility for handling translations
const i18n = {
    currentLocale: 'en',
    supportedLocales: ['en', 'de'], // Add more locales as needed
    translations: {},
    defaultLocale: 'en',
    
    // Initialize the i18n system
    init: function() {
        // Try to load the saved locale or use the browser's language
        const savedLocale = localStorage.getItem('uiLanguage');
        const browserLocale = navigator.language.split('-')[0]; // Get language part only (e.g., 'en' from 'en-US')
        
        if (savedLocale && this.supportedLocales.includes(savedLocale)) {
            this.currentLocale = savedLocale;
        } else if (this.supportedLocales.includes(browserLocale)) {
            // Use browser locale if supported
            this.currentLocale = browserLocale;
        } else {
            // Fall back to English
            this.currentLocale = this.defaultLocale;
        }
        
        // Save the current locale
        localStorage.setItem('uiLanguage', this.currentLocale);
        
        return this;
    },
    
    // Load translations for the current language
    loadTranslations: async function() {
        // If we already have loaded the current locale, no need to reload
        if (this.translations[this.currentLocale]) {
            return this;
        }
        
        try {
            // Simple dynamic import to load the correct language file
            const module = await import(`./${this.currentLocale}.js`);
            this.translations[this.currentLocale] = module.default || window.i18n[this.currentLocale];
        } catch (error) {
            console.error(`Failed to load translations for ${this.currentLocale}:`, error);
            // Fallback to English if loading fails
            if (this.currentLocale !== this.defaultLocale) {
                this.currentLocale = this.defaultLocale;
                return this.loadTranslations();
            }
        }
        
        return this;
    },
    
    // Get a translation by key
    t: function(key) {
        // Make sure the current locale is loaded
        if (!this.translations[this.currentLocale]) {
            console.warn(`Translations for ${this.currentLocale} not loaded yet.`);
            return key;
        }
        
        // Return the translation or the key if not found
        return this.translations[this.currentLocale][key] || this.translations[this.defaultLocale][key] || key;
    },
    
    // Change the current locale
    setLocale: async function(locale) {
        if (!this.supportedLocales.includes(locale)) {
            console.warn(`Locale ${locale} is not supported. Using ${this.defaultLocale} instead.`);
            locale = this.defaultLocale;
        }
        
        this.currentLocale = locale;
        localStorage.setItem('uiLanguage', locale);
        
        // Load translations for the new locale
        await this.loadTranslations();
        
        // Trigger an event so the UI can update
        window.dispatchEvent(new CustomEvent('languageChanged', { detail: { locale } }));
        
        return this;
    },
    
    // Apply translations to all elements with data-i18n attribute
    translatePage: function() {
        document.querySelectorAll('[data-i18n]').forEach(element => {
            const key = element.getAttribute('data-i18n');
            if (key) {
                element.textContent = this.t(key);
            }
        });
        
        // Also translate placeholder attributes
        document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
            const key = element.getAttribute('data-i18n-placeholder');
            if (key) {
                element.setAttribute('placeholder', this.t(key));
            }
        });
        
        // Translate title attributes
        document.querySelectorAll('[data-i18n-title]').forEach(element => {
            const key = element.getAttribute('data-i18n-title');
            if (key) {
                element.setAttribute('title', this.t(key));
            }
        });
        
        return this;
    }
};

// For browser use
window.i18n = i18n;

// For module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = i18n;
} 