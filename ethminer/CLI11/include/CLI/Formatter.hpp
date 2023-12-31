/* 
 * Copyright (C) <2023> Intel Corporation
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, as published
 * by the Free Software Foundation; either version 2 of the License,
 * or (at your option) any later version.
 *  
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *  
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 *  
 * 
 * SPDX-License-Identifier: GPL-2.0-or-later
 * 
 */ 

#pragma once

// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/CLIUtils/CLI11 for details.

#include <string>

#include "CLI/App.hpp"
#include "CLI/FormatterFwd.hpp"

namespace CLI {

inline std::string
Formatter::make_group(std::string group, bool is_positional, std::vector<const Option *> opts) const {
    std::stringstream out;

    out << "\n" << group << ":\n";
    for(const Option *opt : opts) {
        out << make_option(opt, is_positional);
    }

    return out.str();
}

inline std::string Formatter::make_positionals(const App *app) const {
    std::vector<const Option *> opts =
        app->get_options([](const Option *opt) { return !opt->get_group().empty() && opt->get_positional(); });

    if(opts.empty())
        return std::string();
    else
        return make_group(get_label("Positionals"), true, opts);
}

inline std::string Formatter::make_groups(const App *app, AppFormatMode mode) const {
    std::stringstream out;
    std::vector<std::string> groups = app->get_groups();

    // Options
    for(const std::string &group : groups) {
        std::vector<const Option *> opts = app->get_options([app, mode, &group](const Option *opt) {
            return opt->get_group() == group                    // Must be in the right group
                   && opt->nonpositional()                      // Must not be a positional
                   && (mode != AppFormatMode::Sub               // If mode is Sub, then
                       || (app->get_help_ptr() != opt           // Ignore help pointer
                           && app->get_help_all_ptr() != opt)); // Ignore help all pointer
        });
        if(!group.empty() && !opts.empty()) {
            out << make_group(group, false, opts);

            if(group != groups.back())
                out << "\n";
        }
    }

    return out.str();
}

inline std::string Formatter::make_description(const App *app) const {
    std::string desc = app->get_description();

    if(!desc.empty())
        return desc + "\n";
    else
        return "";
}

inline std::string Formatter::make_usage(const App *app, std::string name) const {
    std::stringstream out;

    out << get_label("Usage") << ":" << (name.empty() ? "" : " ") << name;

    std::vector<std::string> groups = app->get_groups();

    // Print an Options badge if any options exist
    std::vector<const Option *> non_pos_options =
        app->get_options([](const Option *opt) { return opt->nonpositional(); });
    if(!non_pos_options.empty())
        out << " [" << get_label("OPTIONS") << "]";

    // Positionals need to be listed here
    std::vector<const Option *> positionals = app->get_options([](const Option *opt) { return opt->get_positional(); });

    // Print out positionals if any are left
    if(!positionals.empty()) {
        // Convert to help names
        std::vector<std::string> positional_names(positionals.size());
        std::transform(positionals.begin(), positionals.end(), positional_names.begin(), [this](const Option *opt) {
            return make_option_usage(opt);
        });

        out << " " << detail::join(positional_names, " ");
    }

    // Add a marker if subcommands are expected or optional
    if(!app->get_subcommands({}).empty()) {
        out << " " << (app->get_require_subcommand_min() == 0 ? "[" : "")
            << get_label(app->get_require_subcommand_max() < 2 || app->get_require_subcommand_min() > 1 ? "SUBCOMMAND"
                                                                                                        : "SUBCOMMANDS")
            << (app->get_require_subcommand_min() == 0 ? "]" : "");
    }

    out << std::endl;

    return out.str();
}

inline std::string Formatter::make_footer(const App *app) const {
    std::string footer = app->get_footer();
    if(!footer.empty())
        return footer + "\n";
    else
        return "";
}

inline std::string Formatter::make_help(const App *app, std::string const &name, AppFormatMode mode) const {

    // This immediately forwards to the make_expanded method. This is done this way so that subcommands can
    // have overridden formatters
    if(mode == AppFormatMode::Sub)
        return make_expanded(app);

    std::stringstream out;

    out << make_description(app);
    out << make_usage(app, name);
    out << make_positionals(app);
    out << make_groups(app, mode);
    out << make_subcommands(app, mode);
    out << make_footer(app);

    return out.str();
}

inline std::string Formatter::make_subcommands(const App *app, AppFormatMode mode) const {
    std::stringstream out;

    std::vector<const App *> subcommands = app->get_subcommands({});

    // Make a list in definition order of the groups seen
    std::vector<std::string> subcmd_groups_seen;
    for(const App *com : subcommands) {
        std::string group_key = com->get_group();
        if(!group_key.empty() &&
           std::find_if(subcmd_groups_seen.begin(), subcmd_groups_seen.end(), [&group_key](std::string const &a) {
               return detail::to_lower(a) == detail::to_lower(group_key);
           }) == subcmd_groups_seen.end())
            subcmd_groups_seen.push_back(group_key);
    }

    // For each group, filter out and print subcommands
    for(const std::string &group : subcmd_groups_seen) {
        out << "\n" << group << ":\n";
        std::vector<const App *> subcommands_group = app->get_subcommands(
            [&group](const App *sub_app) { return detail::to_lower(sub_app->get_group()) == detail::to_lower(group); });
        for(const App *new_com : subcommands_group) {
            if(mode != AppFormatMode::All) {
                out << make_subcommand(new_com);
            } else {
                out << new_com->help(new_com->get_name(), AppFormatMode::Sub);
                out << "\n";
            }
        }
    }

    return out.str();
}

inline std::string Formatter::make_subcommand(const App *sub) const {
    std::stringstream out;
    detail::format_help(out, sub->get_name(), sub->get_description(), column_width_);
    return out.str();
}

inline std::string Formatter::make_expanded(const App *sub) const {
    std::stringstream out;
    out << sub->get_name() << "\n";

    out << make_description(sub);
    out << make_positionals(sub);
    out << make_groups(sub, AppFormatMode::Sub);
    out << make_subcommands(sub, AppFormatMode::Sub);

    // Drop blank spaces
    std::string tmp = detail::find_and_replace(out.str(), "\n\n", "\n");
    tmp = tmp.substr(0, tmp.size() - 1); // Remove the final '\n'

    // Indent all but the first line (the name)
    return detail::find_and_replace(tmp, "\n", "\n  ") + "\n";
}

inline std::string Formatter::make_option_name(const Option *opt, bool is_positional) const {
    if(is_positional)
        return opt->get_name(true, false);
    else
        return opt->get_name(false, true);
}

inline std::string Formatter::make_option_opts(const Option *opt) const {
    std::stringstream out;

    if(opt->get_type_size() != 0) {
        if(!opt->get_type_name().empty())
            out << " " << get_label(opt->get_type_name());
        if(!opt->get_defaultval().empty())
            out << "=" << opt->get_defaultval();
        if(opt->get_expected() > 1)
            out << " x " << opt->get_expected();
        if(opt->get_expected() == -1)
            out << " ...";
        if(opt->get_required())
            out << " " << get_label("REQUIRED");
    }
    if(!opt->get_envname().empty())
        out << " (" << get_label("Env") << ":" << opt->get_envname() << ")";
    if(!opt->get_needs().empty()) {
        out << " " << get_label("Needs") << ":";
        for(const Option *op : opt->get_needs())
            out << " " << op->get_name();
    }
    if(!opt->get_excludes().empty()) {
        out << " " << get_label("Excludes") << ":";
        for(const Option *op : opt->get_excludes())
            out << " " << op->get_name();
    }
    return out.str();
}

inline std::string Formatter::make_option_desc(const Option *opt) const { return opt->get_description(); }

inline std::string Formatter::make_option_usage(const Option *opt) const {
    // Note that these are positionals usages
    std::stringstream out;
    out << make_option_name(opt, true);

    if(opt->get_expected() > 1)
        out << "(" << std::to_string(opt->get_expected()) << "x)";
    else if(opt->get_expected() < 0)
        out << "...";

    return opt->get_required() ? out.str() : "[" + out.str() + "]";
}

} // namespace CLI
