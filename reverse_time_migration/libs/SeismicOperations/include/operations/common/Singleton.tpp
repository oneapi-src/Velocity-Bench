//
// Created by amr on 03/01/2021.
//

#ifndef OPERATIONS_LIB_BASE_SINGLETON_HPP
#define OPERATIONS_LIB_BASE_SINGLETON_HPP

#include <iostream>


namespace operations {
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
                if (INSTANCE == nullptr) {
                    INSTANCE = new T;
                }
                return INSTANCE;
            }

            /**
             * @brief Destroy the active instance of the singleton.
             */
            static T *Kill() {
                if (INSTANCE != nullptr) {
                    delete INSTANCE;
                    INSTANCE = nullptr;
                }
                return INSTANCE;
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
            static T *INSTANCE;
        };

        template<typename T> T *Singleton<T>::INSTANCE = nullptr;

    }//namespace common
}//namespace operations

#endif //OPERATIONS_LIB_BASE_SINGLETON_HPP
