#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/lac/sparse_direct.h>
#include <iostream>
#include <fstream>
#include <sstream>

namespace Step22
{
    using namespace dealii;
    using namespace numbers;
    
    namespace data
    {
        const double top = 1.0;
        const double bottom = 0.0;
        const double left = 0.0;
        const double right = PI;
        const double phi = 0.7;
        
        const double timestep_size = 0.1;
        const double total_timesteps = 1;
        
        const int refinement_level = 1;
        int timestep_number;
        double time;
    }
    using namespace data;
    
    template <int dim>
    class StokesProblem
    {
    public:
        StokesProblem (const unsigned int degree);
        void run ();
    private:
        void initial_grid ();
        void setup_dofs ();
        void apply_BCs ();
        void assemble_system ();
        void solve ();
        void output_results () const;
        void compute_errors ();
        void move_mesh ();
        void print_mesh ();
        void add_sediment ();
        void print_mesh_info (const Triangulation<dim> &tria,
                              const std::string        &filename);
        const unsigned int   degree;
        Triangulation<dim>   triangulation;
        FESystem<dim>        fe;
        DoFHandler<dim>      dof_handler;
        ConstraintMatrix     constraints;
        BlockSparsityPattern      sparsity_pattern;
        BlockSparseMatrix<double> system_matrix;
        BlockVector<double> solution;
        BlockVector<double> system_rhs;
        Vector<double>            displacement;
        Triangulation<dim>  sediment_tria;
        Triangulation<dim>  final_tria;
    };
    
    template <int dim>
    class RockExactSolution : public Function<dim>
    {
    public:
        RockExactSolution () : Function<dim>(dim+1) {}
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &value) const;
    };
    
    template <int dim>
    void
    RockExactSolution<dim>::vector_value (const Point<dim> &p,
                                          Vector<double>   &values) const
    {
        values(0) = sin(p[0]);
        values(1) = -p[0]*p[1]*p[1];
        values(2) = p[0]*p[1];
    }
    
    // Extra terms you get on the right hand side from manufactured solutions
    template <int dim>
    class ExtraRHSRock : public Function<dim>
    {
    public:
        ExtraRHSRock () : Function<dim>(dim+1) {}
        virtual void vector_value (const Point<dim> &p, Vector<double>   &value) const;
    };
    
    template <int dim>
    void
    ExtraRHSRock<dim>::vector_value (const Point<dim> &p, Vector<double>   &values) const
    {
        values(0) = (1-phi)*(-2*sin(p[0]) - 2*p[1] - p[1]) ;
        values(1) = (1-phi)*( -4*p[0] - p[0]);
        values(2) = (1-phi)*( cos(p[0])-2*p[0]*p[1] );
    }
    template <int dim>
    class ExactSolution_phi : public Function<dim>
    {
    public:
        ExactSolution_phi () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    template <int dim>
    double ExactSolution_phi<dim>::value (const Point<dim>  &p,
                                          const unsigned int /*component*/) const
    {
        return  phi + p[1]*0;
    }

    template <int dim>
    StokesProblem<dim>::StokesProblem (const unsigned int degree)
    :
    degree (degree),
    triangulation (Triangulation<dim>::maximum_smoothing),
    fe (FE_Q<dim>(degree+1), dim,
        FE_Q<dim>(degree), 1),
    dof_handler (triangulation)
    {}
    
    template <int dim>
    void StokesProblem<dim>::initial_grid ()
    {
        
        std::vector<unsigned int> subdivisions (dim, 1);
        subdivisions[0] = 4;
        
        const Point<dim> bottom_left = (dim == 2 ?
                                        Point<dim>(data::left,data::bottom) :
                                        Point<dim>(-2,0,-1));
        const Point<dim> top_right   = (dim == 2 ?
                                        Point<dim>(data::right,data::top) :
                                        Point<dim>(0,1,0));
        
        GridGenerator::subdivided_hyper_rectangle (triangulation,
                                                   subdivisions, bottom_left, top_right);
        
        for (typename Triangulation<dim>::active_cell_iterator
             cell = triangulation.begin_active();
             cell != triangulation.end(); ++cell)
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->center()[dim-1] == data::top)
                    cell->face(f)->set_all_boundary_ids(1);
                else if (cell->face(f)->center()[dim-1] == data::bottom)
                    cell->face(f)->set_all_boundary_ids(2);
        
        print_mesh_info(triangulation, "first_grid.eps");

        
        triangulation.refine_global (2);
        
        print_mesh_info(triangulation, "after_refine.eps");
//
//
//        for (typename Triangulation<dim>::active_cell_iterator
//             cell = triangulation.begin_active();
//             cell != triangulation.end(); ++cell)
//        {
//            cell->set_coarsen_flag();
//        }
//
//        triangulation.execute_coarsening_and_refinement ();
//
//        print_mesh_info(triangulation, "after_coarsen.eps");

    }
    
    template <int dim>
    void StokesProblem<dim>::setup_dofs ()
    {
        system_matrix.clear ();
        dof_handler.distribute_dofs (fe);
        DoFRenumbering::Cuthill_McKee (dof_handler);
        std::vector<unsigned int> block_component (dim+1,0);
        block_component[dim] = 1;
        DoFRenumbering::component_wise (dof_handler, block_component);
        
        std::vector<types::global_dof_index> dofs_per_block (2);
        DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
        const unsigned int n_u = dofs_per_block[0],
        n_p = dofs_per_block[1];
        std::cout << "   Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl
        << "   Number of degrees of freedom: "
        << dof_handler.n_dofs()
        << " (" << n_u << '+' << n_p << ')'
        << std::endl;
        
        {
            BlockDynamicSparsityPattern dsp (2,2);
            dsp.block(0,0).reinit (n_u, n_u);
            dsp.block(1,0).reinit (n_p, n_u);
            dsp.block(0,1).reinit (n_u, n_p);
            dsp.block(1,1).reinit (n_p, n_p);
            dsp.collect_sizes();
            DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
            sparsity_pattern.copy_from (dsp);
        }
        
        system_matrix.reinit (sparsity_pattern);
        solution.reinit (2);
        solution.block(0).reinit (n_u);
        solution.block(1).reinit (n_p);
        solution.collect_sizes ();
        system_rhs.reinit (2);
        system_rhs.block(0).reinit (n_u);
        system_rhs.block(1).reinit (n_p);
        system_rhs.collect_sizes ();
    }
    
    template <int dim>
    void StokesProblem<dim>::apply_BCs ()
    {
        {
            constraints.clear ();
            
            FEValuesExtractors::Vector velocities(0);
            FEValuesExtractors::Scalar pressure (dim);
            
            DoFTools::make_hanging_node_constraints (dof_handler,
                                                     constraints);
            
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      1,
                                                      RockExactSolution<dim>(),
                                                      constraints,
                                                      fe.component_mask(velocities));
            
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      2,
                                                      RockExactSolution<dim>(),
                                                      constraints,
                                                      fe.component_mask(velocities));
            
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      0,
                                                      RockExactSolution<dim>(),
                                                      constraints,
                                                      fe.component_mask(velocities));
        }
        constraints.close ();
    }
    
    template <int dim>
    void StokesProblem<dim>::assemble_system ()
    {
        system_matrix=0;
        system_rhs=0;
        QGauss<dim>   quadrature_formula(degree+2);
        
        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    |
                                 update_quadrature_points  |
                                 update_JxW_values |
                                 update_gradients);
        const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        
        const ExtraRHSRock<dim>          right_hand_side;
        std::vector<Vector<double> >      rhs_values (n_q_points,
                                                      Vector<double>(dim+1));
        const ExactSolution_phi<dim>    phi_solution;
        std::vector<double>             phi_values(n_q_points);

        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        std::vector<SymmetricTensor<2,dim> > symgrad_phi_u (dofs_per_cell);
        std::vector<double>                  div_phi_u   (dofs_per_cell);
        std::vector<double>                  phi_p       (dofs_per_cell);
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            fe_values.reinit (cell);
            local_matrix = 0;
            local_rhs = 0;
            right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                              rhs_values);
            phi_solution.value_list(fe_values.get_quadrature_points(), phi_values);
            
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                for (unsigned int k=0; k<dofs_per_cell; ++k)
                {
                    symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);
                    div_phi_u[k]     = fe_values[velocities].divergence (k, q);
                    phi_p[k]         = fe_values[pressure].value (k, q);
                }
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    for (unsigned int j=0; j<=i; ++j)
                    { // matrix assembly
                        local_matrix(i,j) += (-2 * (1-phi_values[q])*(symgrad_phi_u[i] * symgrad_phi_u[j])
                                              + (1-phi_values[q])*div_phi_u[i] * phi_p[j]
                                              + (1-phi_values[q])*phi_p[i] * div_phi_u[j])
                        * fe_values.JxW(q);
                    }
                    
                    const unsigned int component_i =
                    fe.system_to_component_index(i).first;
                    
                    // Add the extra terms on RHS
                    local_rhs(i) +=
                    fe_values.shape_value(i,q) *
                    rhs_values[q](component_i) *
                    fe_values.JxW(q);
                }
            }
        
            
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=i+1; j<dofs_per_cell; ++j)
                    local_matrix(i,j) = local_matrix(j,i);
            cell->get_dof_indices (local_dof_indices);
            
            constraints.distribute_local_to_global (local_matrix, local_rhs,
                                                    local_dof_indices,
                                                    system_matrix, system_rhs);
        }
        

        std::map<types::global_dof_index, double> vr_determination;
        {
            types::global_dof_index n_dofs = dof_handler.n_dofs();
            std::vector<bool> componentVector(dim + 1, false);
            componentVector[dim] = true;
            
            std::vector<bool> selected_dofs(n_dofs);
            std::set< types::boundary_id > boundary_ids;
            boundary_ids.insert(2);
            
            DoFTools::extract_boundary_dofs(dof_handler, ComponentMask(componentVector),
                                            selected_dofs, boundary_ids);
            
            for (types::global_dof_index i = 0; i < n_dofs; i++)
            {
                if (selected_dofs[i]) vr_determination[i] = 0.0;
            }
        }
        
        MatrixTools::apply_boundary_values(vr_determination,
                                           system_matrix, solution, system_rhs);
    }
    
    template <int dim>
    void StokesProblem<dim>::solve ()
    {
        SparseDirectUMFPACK  A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult (solution, system_rhs);
        constraints.distribute (solution);
    }
    
    template <int dim>
    void StokesProblem<dim>::compute_errors ()
    {
        {
            const ComponentSelectFunction<dim> pressure_mask (dim, dim+1);
            const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim+1);
            RockExactSolution<dim> exact_solution;
            
            Vector<double> cellwise_errors (triangulation.n_active_cells());
            
            QTrapez<1>     q_trapez;
            QIterated<dim> quadrature (q_trapez, degree+2);
            
            VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                               cellwise_errors, quadrature,
                                               VectorTools::L2_norm,
                                               &pressure_mask);
            const double p_l2_error = cellwise_errors.l2_norm();
            
            VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                               cellwise_errors, quadrature,
                                               VectorTools::L2_norm,
                                               &velocity_mask);
            const double u_l2_error = cellwise_errors.l2_norm();
            std::cout << "   Errors: ||e_pr||_L2  = " << p_l2_error
            << ",  " << std::endl << "           ||e_vr||_L2  = " << u_l2_error
            << std::endl;
        }
        
    }
    
    
    template <int dim>
    void
    StokesProblem<dim>::output_results ()  const
    {
        std::vector<std::string> solution_names (dim, "velocity");
        solution_names.push_back ("pressure");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation
        (dim, DataComponentInterpretation::component_is_part_of_vector);
        data_component_interpretation
        .push_back (DataComponentInterpretation::component_is_scalar);
        DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (solution, solution_names,
                                  DataOut<dim>::type_dof_data,
                                  data_component_interpretation);
        data_out.build_patches ();
        std::ostringstream filename;
        filename << "solution"+ Utilities::int_to_string(timestep_number, 3) + ".vtk";
        std::ofstream output (filename.str().c_str());
        data_out.write_vtk (output);

    }
    
    template <int dim>
    void StokesProblem<dim>::move_mesh ()
    {
        std::cout << "   Moving mesh..." << std::endl;
        
        std::vector<bool> vertex_touched (triangulation.n_vertices(),
                                          false);
        
        for (typename DoFHandler<dim>::active_cell_iterator
             cell = dof_handler.begin_active ();
             cell != dof_handler.end(); ++cell)
            for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                if (vertex_touched[cell->vertex_index(v)] == false)
                {
                    vertex_touched[cell->vertex_index(v)] = true;
                    Point<dim> vertex_displacement;
                    for (unsigned int d=0; d<dim; ++d)
                    {vertex_displacement[d]
                        = solution.block(0)(cell->vertex_dof_index(v,d));
                    }
                    cell->vertex(v) += vertex_displacement*timestep_size;
                    //                    std::cout << cell->diameter() << std::endl;
                }
    }
    
    template <int dim>
    void StokesProblem<dim>::print_mesh ()
    {
        std::ofstream out ("grid-"
                           + Utilities::int_to_string(timestep_number, 3) +
                           ".eps");
        GridOut grid_out;
        grid_out.write_eps (triangulation, out);
    }
    
    template <int dim>
    void StokesProblem<dim>::add_sediment ()
    {
        std::vector<Point<2>> vertex_points;
        vertex_points.push_back (Point<dim> (0,1.));
        unsigned int p = 0;

        for (typename Triangulation<dim>::active_cell_iterator
             cell = triangulation.begin_active();
             cell != triangulation.end(); ++cell)
        {
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            {
                if (cell->face(f)->boundary_id() == 1)
                {
                    Point<2> points = cell->face(f)->vertex(1);
//                    std::cout << points << std::endl;
                    vertex_points.push_back (points);
                    ++p;
                }
            }
        }
        
//        std::vector<Point<2>> new_vertices;
        const double end_z_point = vertex_points[vertex_points.size()-1][1];
//
//        new_vertices.push_back (Point<dim> (0, 1.));
//        new_vertices.push_back (Point<dim> (PI, end_z_point));
//        new_vertices.push_back (Point<dim> (0, 1.+0.2));
//        new_vertices.push_back (Point<dim> (PI, end_z_point+0.2));
//        const unsigned int vertex_length = vertex_points.size();

        // Output originsl boundary 1 vertices
        for (unsigned int i=0; i<=vertex_points.size()-1; ++i)
            std::cout << vertex_points[i] << std::endl;

        std::cout << "Number of vertex points: " << vertex_points.size() <<std::endl;

        for (unsigned int i=0; i<17; ++i)
        {
            vertex_points.push_back (Point<dim> (vertex_points[i][0], vertex_points[i][1]+0.2));
        }

        // output vertex points of new deposited layer
        for (unsigned int i=0; i<=33; ++i)
            std::cout << vertex_points[i] << std::endl;

//
//        // numbering will go from 0 to vertex_length-1
        unsigned int cell_vertices[][GeometryInfo<dim>::vertices_per_cell]= {{0, 1, 17,18}, {1,2,18,19},{2,3,19,20},{3,4,20,21},{4,5,21,22},{5,6,22,23},{6,7,23,24}
            ,{7,8,24,25},{8,9,25,26},{9,10,26,27},{10,11,27,28},{11,12,28,29},{12,13,29,30},{13,14,30,31}, {14,15,31,32}, {15,16,32,33}
        };

        const unsigned int
        n_cells = sizeof(cell_vertices) / sizeof(cell_vertices[0]);
        std::vector<CellData<dim> > cells (n_cells, CellData<dim>());
        for (unsigned int i=0; i<n_cells; ++i)
        {
            for (unsigned int j=0;
                 j<GeometryInfo<dim>::vertices_per_cell;
                 ++j)
                cells[i].vertices[j] = cell_vertices[i][j];
            cells[i].material_id = 0;
        }

        sediment_tria.create_triangulation (vertex_points,
                                    cells,
                                    SubCellData());

        print_mesh_info(sediment_tria, "sediment_grid_refined.eps");
        
    
//        for (typename Triangulation<dim>::active_cell_iterator
//             cell = triangulation.begin_active();
//             cell != triangulation.end(); ++cell)
//        {
//            cell->set_coarsen_flag();
//        }
//
//        triangulation.execute_coarsening_and_refinement ();

        print_mesh_info(triangulation, "initial_grid.eps");

//        GridGenerator::create_union_triangulation (sediment_tria, triangulation, final_tria);
//                                             print_mesh_info(final_tria, "final_grid.eps");
        {
            Triangulation<dim> merged_tria;
        std::vector<Point<2>> merged_vertices;
        merged_vertices.push_back (Point<dim> (0, 0.));
        merged_vertices.push_back (Point<dim> (PI, 0.));
        merged_vertices.push_back (Point<dim> (0, 1.+0.2));
        merged_vertices.push_back (Point<dim> (PI, end_z_point+0.2));
        
        unsigned int merged_cell_vertices[][GeometryInfo<dim>::vertices_per_cell]= {{0, 1, 2, 3}
        };
        
//        const unsigned int
//        merged_n_cells = sizeof(merged_cell_vertices) / sizeof(merged_cell_vertices[0]);
//        std::vector<CellData<dim> > merged_cells (merged_n_cells, CellData<dim>());
//        for (unsigned int i=0; i<merged_n_cells; ++i)
//        {
//            for (unsigned int j=0;
//                 j<GeometryInfo<dim>::vertices_per_cell;
//                 ++j)
//                merged_cells[i].vertices[j] = merged_cell_vertices[i][j];
//            merged_cells[i].material_id = 0;
//        }
//
//        merged_tria.create_triangulation (merged_vertices,
//                                            merged_cells,
//                                            SubCellData());
////            merged_tria.refine_global(1);
////
////            for (typename Triangulation<dim>::active_cell_iterator
////                 cell = merged_tria.begin_active();
////                 cell != merged_tria.end(); ++cell)
////            {
////                cell->set_refine_flag(RefinementCase<dim>::cut_x);
//////                std::cout << "refine_flag" << std::endl;
////            }
////            merged_tria.execute_coarsening_and_refinement ();
////            for (typename Triangulation<dim>::active_cell_iterator
////                 cell = merged_tria.begin_active();
////                 cell != merged_tria.end(); ++cell)
////            {
////                cell->set_refine_flag(RefinementCase<dim>::cut_x);
////                //                std::cout << "refine_flag" << std::endl;
////            }
////            merged_tria.execute_coarsening_and_refinement ();
////            merged_tria.refine_global(1);
//
//        print_mesh_info(merged_tria, "merged_grid_before_refine.eps");
        }
        
        
    }
    
    template <int dim>
    void StokesProblem<dim>::print_mesh_info(const Triangulation<dim> &tria,
                         const std::string        &filename)
    {
        std::cout << "Mesh info:" << std::endl
        << " dimension: " << dim << std::endl
        << " no. of cells: " << tria.n_active_cells() << std::endl
        << " no. of vertices: " << tria.n_vertices() << std::endl;
        {
            std::map<unsigned int, unsigned int> boundary_count;
            typename Triangulation<dim>::active_cell_iterator
            cell = tria.begin_active(),
            endc = tria.end();
            for (; cell!=endc; ++cell)
            {
                for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
                {
                    if (cell->face(face)->at_boundary())
                        boundary_count[cell->face(face)->boundary_id()]++;
                }
            }
            std::cout << " boundary indicators: ";
            for (std::map<unsigned int, unsigned int>::iterator it=boundary_count.begin();
                 it!=boundary_count.end();
                 ++it)
            {
                std::cout << it->first << "(" << it->second << " times) ";
            }
            std::cout << std::endl;
        }
        std::ofstream out (filename.c_str());
        GridOut grid_out;
        grid_out.write_eps (tria, out);
        std::cout << " written to " << filename
        << std::endl
        << std::endl;
    }
    
    template <int dim>
    void StokesProblem<dim>::run ()
    {
        initial_grid ();
        setup_dofs ();
        
        timestep_number = 0;
        time = 0.0;
        apply_BCs ();
        std::cout << "   Assembling..." << std::endl << std::flush;
        assemble_system ();
        std::cout << "   Solving..." << std::endl << std::flush;
        solve ();
        output_results ();
        compute_errors ();
        print_mesh ();
        move_mesh ();

        add_sediment ();

        
//        while (timestep_number < total_timesteps)
//        {
//            time += timestep_size;
//            ++timestep_number;
//
//            apply_BCs ();
//            std::cout << "   Assembling..." << std::endl << std::flush;
//            assemble_system ();
//            std::cout << "   Solving..." << std::endl << std::flush;
//            solve ();
//            output_results ();
//            compute_errors ();
//            print_mesh ();
//            move_mesh ();
//        }
    }
}
int main ()
{
    try
    {
        using namespace dealii;
        using namespace Step22;
        StokesProblem<2> flow_problem(1);
        flow_problem.run ();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Exception on processing: " << std::endl
        << exc.what() << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Unknown exception!" << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }
    return 0;
}

