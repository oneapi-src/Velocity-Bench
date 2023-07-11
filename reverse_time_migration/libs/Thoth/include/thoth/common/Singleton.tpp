//
// Created by zeyad-osama on 02/11/2020.
//

#ifndef THOTH_COMMON_SINGLETON_TPP
#define THOTH_COMMON_SINGLETON_TPP

namespace thoth {
    namespace common {
        /**
         * @brief Template class for any singleton class to inherit.
         * @tparam[in] T The type for the singleton pattern to be applied to.
         */
        template<typename T>
        class Singleton {
        public:
            /**
             * @brief Static function to obtain the instance of the singleton to use.
             * @return An instance of the singleton.
             */
            static T *GetInstance() {
                if (_INSTANCE == nullptr) {
                    _INSTANCE = new T;
                }
                return _INSTANCE;
            }

            /**
             * @brief Destroy the active instance of the singleton.
             */
            static void Kill() {
                if (_INSTANCE != nullptr) {
                    delete _INSTANCE;
                    _INSTANCE = nullptr;
                }
            }

        protected:
            /**
             * @brief Default constructor.
             */
            Singleton() = default;

            /**
             * @brief Default Destructor.
             */
            ~Singleton() = default;

        private:
            /// The instance to utilize in the class.
            static T *_INSTANCE;
        };

        template<typename T> T *Singleton<T>::_INSTANCE = nullptr;
    }
} //thoth::common

#endif //THOTH_COMMON_SINGLETON_TPP
