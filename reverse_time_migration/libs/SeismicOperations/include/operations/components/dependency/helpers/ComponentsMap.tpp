//
// Created by zeyad-osama on 27/09/2020.
//

#ifndef OPERATIONS_LIB_COMPONENTS_DEPENDENT_COMPONENTS_MAP_TPP
#define OPERATIONS_LIB_COMPONENTS_DEPENDENT_COMPONENTS_MAP_TPP

#include <operations/components/dependents/interface/DependentComponent.hpp>
#include <operations/exceptions/Exceptions.h>

#include <vector>
#include <map>

namespace operations {
    namespace helpers {

        template<typename T>
        class ComponentsMap {
        public:
            ~ComponentsMap() = default;

            void Set(uint key, T *apDependentComponent) {
                this->mComponentsMap[key] = apDependentComponent;
            }

            T *Get(uint key) {
                if (this->mComponentsMap.find(key) ==
                    this->mComponentsMap.end()) {
                    throw exceptions::NotFoundException();
                }
                return this->mComponentsMap[key];
            }

            std::vector<T *> ExtractValues() {
                std::vector<T *> values;
                for (auto const &dependent_components : this->mComponentsMap) {
                    values.push_back(dependent_components.second);
                }
                return values;
            }

        private:
            std::map<uint, T *> mComponentsMap;
        };

/*
 * Indices.
 */
#define MEMORY_HANDLER 0

#define MODEL_HANDLER 1
#define COMPUTATION_KERNEL 2
#define MIGRATION_ACCOMMODATOR 3
#define BOUNDARY_MANAGER 4
#define SOURCE_INJECTOR 5
#define TRACE_MANAGER 6
#define RAY_TRACER 7
#define RESIDUAL_MANAGER 8
#define STOPPAGE_CRITERIA 9
#define MODEL_UPDATER 10
#define FORWARD_COLLECTOR 11
#define TRACE_WRITER 12
#define MODELLING_CONFIG_PARSER 13

    }//namespace helpers
}//namespace operations

#endif //OPERATIONS_LIB_COMPONENTS_DEPENDENT_COMPONENTS_MAP_TPP
